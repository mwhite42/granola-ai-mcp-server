"""Granola MCP Server implementation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import zoneinfo
import time

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    TextContent,
    Tool,
)

from .models import CacheData, MeetingMetadata, MeetingDocument, MeetingTranscript


class GranolaMCPServer:
    """Granola MCP Server for meeting intelligence queries."""

    def __init__(
        self, cache_path: Optional[str] = None, timezone: Optional[str] = None
    ):
        """Initialize the Granola MCP server."""
        if cache_path is None:
            cache_path = os.path.expanduser(
                "~/Library/Application Support/Granola/cache-v3.json"
            )

        self.cache_path = cache_path
        self.server = Server("granola-mcp-server")
        self.cache_data: Optional[CacheData] = None

        # Set up timezone handling
        if timezone:
            self.local_timezone = zoneinfo.ZoneInfo(timezone)
        else:
            # Auto-detect local timezone
            self.local_timezone = self._detect_local_timezone()

        self._setup_handlers()

    def _detect_local_timezone(self):
        """Detect the local timezone."""
        try:
            # Try to get system timezone
            if hasattr(time, "tzname") and time.tzname:
                # Convert system timezone to zoneinfo
                # Common mappings for US timezones
                tz_mapping = {
                    "EST": "America/New_York",
                    "EDT": "America/New_York",
                    "CST": "America/Chicago",
                    "CDT": "America/Chicago",
                    "MST": "America/Denver",
                    "MDT": "America/Denver",
                    "PST": "America/Los_Angeles",
                    "PDT": "America/Los_Angeles",
                }

                current_tz = time.tzname[time.daylight]
                if current_tz in tz_mapping:
                    return zoneinfo.ZoneInfo(tz_mapping[current_tz])

            # Fallback: try to detect from system offset
            local_offset = time.timezone if not time.daylight else time.altzone
            hours_offset = -local_offset // 3600

            # Common US timezone mappings by offset
            offset_mapping = {
                -8: "America/Los_Angeles",  # PST
                -7: "America/Denver",  # MST
                -6: "America/Chicago",  # CST
                -5: "America/New_York",  # EST
                -4: "America/New_York",  # EDT (during daylight saving)
            }

            if hours_offset in offset_mapping:
                return zoneinfo.ZoneInfo(offset_mapping[hours_offset])

        except Exception as e:
            print(f"Error detecting timezone: {e}")

        # Ultimate fallback to Eastern Time (common for US business)
        return zoneinfo.ZoneInfo("America/New_York")

    def _convert_to_local_time(self, utc_datetime: datetime) -> datetime:
        """Convert UTC datetime to local timezone."""
        if utc_datetime.tzinfo is None:
            # Assume UTC if no timezone info
            utc_datetime = utc_datetime.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

        return utc_datetime.astimezone(self.local_timezone)

    def _format_local_time(self, utc_datetime: datetime) -> str:
        """Format datetime in local timezone for display."""
        local_dt = self._convert_to_local_time(utc_datetime)
        return local_dt.strftime("%Y-%m-%d %H:%M")

    def _setup_handlers(self):
        """Set up MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="list_meetings",
                    description="List meetings for today, a specific date, or a date range. Returns title, time, and participants (no transcript).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date to list meetings for (YYYY-MM-DD format). Use 'today', 'yesterday', or a specific date. If not provided, defaults to today.",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date for a date range (YYYY-MM-DD format). Use with end_date for range queries.",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date for a date range (YYYY-MM-DD format). Use with start_date for range queries.",
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="search_meetings",
                    description="Search meetings by title, content, or participants",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for meetings",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_meeting_details",
                    description="Get detailed information about a specific meeting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "meeting_id": {
                                "type": "string",
                                "description": "Meeting ID to retrieve details for",
                            }
                        },
                        "required": ["meeting_id"],
                    },
                ),
                Tool(
                    name="get_meeting_transcript",
                    description="Get transcript for a specific meeting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "meeting_id": {
                                "type": "string",
                                "description": "Meeting ID to get transcript for",
                            }
                        },
                        "required": ["meeting_id"],
                    },
                ),
                Tool(
                    name="get_meeting_documents",
                    description="Get documents associated with a meeting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "meeting_id": {
                                "type": "string",
                                "description": "Meeting ID to get documents for",
                            }
                        },
                        "required": ["meeting_id"],
                    },
                ),
                Tool(
                    name="analyze_meeting_patterns",
                    description="Analyze patterns across multiple meetings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern_type": {
                                "type": "string",
                                "description": "Type of pattern to analyze (topics, participants, frequency)",
                                "enum": ["topics", "participants", "frequency"],
                            },
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start_date": {"type": "string", "format": "date"},
                                    "end_date": {"type": "string", "format": "date"},
                                },
                                "description": "Optional date range for analysis",
                            },
                        },
                        "required": ["pattern_type"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            await self._ensure_cache_loaded()

            if name == "list_meetings":
                return await self._list_meetings(
                    date_str=arguments.get("date"),
                    start_date_str=arguments.get("start_date"),
                    end_date_str=arguments.get("end_date"),
                )
            elif name == "search_meetings":
                return await self._search_meetings(
                    query=arguments["query"], limit=arguments.get("limit", 10)
                )
            elif name == "get_meeting_details":
                return await self._get_meeting_details(arguments["meeting_id"])
            elif name == "get_meeting_transcript":
                return await self._get_meeting_transcript(arguments["meeting_id"])
            elif name == "get_meeting_documents":
                return await self._get_meeting_documents(arguments["meeting_id"])
            elif name == "analyze_meeting_patterns":
                return await self._analyze_meeting_patterns(
                    pattern_type=arguments["pattern_type"],
                    date_range=arguments.get("date_range"),
                )
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _ensure_cache_loaded(self):
        """Ensure cache data is loaded."""
        if self.cache_data is None:
            await self._load_cache()

    async def _load_cache(self):
        """Load and parse Granola cache data."""
        try:
            cache_path = Path(self.cache_path)
            if not cache_path.exists():
                self.cache_data = CacheData()
                return

            with open(cache_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Handle Granola's nested cache structure
            if "cache" in raw_data and isinstance(raw_data["cache"], str):
                # Cache data is stored as a JSON string inside the 'cache' key
                actual_data = json.loads(raw_data["cache"])
                if "state" in actual_data:
                    raw_data = actual_data["state"]
                else:
                    raw_data = actual_data

            self.cache_data = await self._parse_cache_data(raw_data)

        except Exception as e:
            self.cache_data = CacheData()
            print(f"Error loading cache: {e}")

    async def _parse_cache_data(self, raw_data: Dict[str, Any]) -> CacheData:
        """Parse raw cache data into structured models."""
        cache_data = CacheData()

        # Parse Granola documents (which are meetings)
        if "documents" in raw_data:
            for meeting_id, meeting_data in raw_data["documents"].items():
                try:
                    # Extract participants from people object
                    participants = []
                    people_data = meeting_data.get("people")
                    if people_data:
                        if isinstance(people_data, dict):
                            # New Granola format: {creator: {...}, attendees: [...]}
                            # Add creator if present
                            creator = people_data.get("creator")
                            if creator and isinstance(creator, dict):
                                creator_name = creator.get("name")
                                if creator_name:
                                    participants.append(creator_name)

                            # Add attendees
                            attendees = people_data.get("attendees")
                            if attendees and isinstance(attendees, list):
                                for attendee in attendees:
                                    if isinstance(attendee, dict):
                                        # Try name first, then email as fallback
                                        name = attendee.get("name")
                                        if name and name != "Unknown":
                                            participants.append(name)
                                        elif not name or name == "Unknown":
                                            # Use email prefix as name if no name
                                            email = attendee.get("email", "")
                                            if email:
                                                participants.append(email.split("@")[0])
                        elif isinstance(people_data, list):
                            # Legacy format: list of people objects
                            participants = [
                                person.get("name", "")
                                for person in people_data
                                if person.get("name")
                            ]

                    # Parse creation date
                    created_at = meeting_data.get("created_at")
                    if created_at:
                        # Handle Granola's ISO format
                        if created_at.endswith("Z"):
                            created_at = created_at[:-1] + "+00:00"
                        naive_date = datetime.fromisoformat(created_at)
                        # Ensure timezone-aware datetime (assume UTC if naive)
                        if naive_date.tzinfo is None:
                            meeting_date = naive_date.replace(
                                tzinfo=zoneinfo.ZoneInfo("UTC")
                            )
                        else:
                            meeting_date = naive_date
                    else:
                        meeting_date = datetime.now(zoneinfo.ZoneInfo("UTC"))

                    title = meeting_data.get("title")
                    if title is None or title == "":
                        title = "Untitled Meeting"

                    metadata = MeetingMetadata(
                        id=meeting_id,
                        title=title or "Untitled Meeting",
                        date=meeting_date,
                        duration=None,  # Granola doesn't store duration in this format
                        participants=participants,
                        meeting_type=meeting_data.get("type", "meeting"),
                        platform=None,  # Not stored in Granola cache
                    )
                    cache_data.meetings[meeting_id] = metadata
                except Exception as e:
                    print(f"Error parsing meeting {meeting_id}: {e}")

        # Parse Granola transcripts (list format)
        if "transcripts" in raw_data:
            for transcript_id, transcript_data in raw_data["transcripts"].items():
                try:
                    # Use transcript_id as meeting_id (they match in Granola)
                    meeting_id = transcript_id

                    # Extract transcript content and speakers
                    content_parts = []
                    speakers_set = set()

                    if isinstance(transcript_data, list):
                        # Granola format: list of speech segments
                        for segment in transcript_data:
                            if isinstance(segment, dict) and "text" in segment:
                                text = segment["text"].strip()
                                if text:
                                    content_parts.append(text)

                                # Extract speaker info if available
                                if "source" in segment:
                                    speakers_set.add(segment["source"])

                    elif isinstance(transcript_data, dict):
                        # Fallback: dict format (legacy or different structure)
                        if "content" in transcript_data:
                            content_parts.append(transcript_data["content"])
                        elif "text" in transcript_data:
                            content_parts.append(transcript_data["text"])
                        elif "transcript" in transcript_data:
                            content_parts.append(transcript_data["transcript"])

                        # Extract speakers if available
                        if "speakers" in transcript_data:
                            speakers_set.update(transcript_data["speakers"])

                    # Combine all content and create transcript
                    if content_parts:
                        full_content = " ".join(content_parts)
                        speakers_list = list(speakers_set) if speakers_set else []

                        transcript = MeetingTranscript(
                            meeting_id=meeting_id,
                            content=full_content,
                            speakers=speakers_list,
                            language=None,  # Not typically stored in segment format
                            confidence=None,  # Would need to be calculated from segments
                        )
                        cache_data.transcripts[meeting_id] = transcript

                except Exception as e:
                    print(f"Error parsing transcript {transcript_id}: {e}")

        # Extract document content from Granola documents
        document_panels = raw_data.get("documentPanels", {})
        parse_panels = os.getenv("GRANOLA_PARSE_PANELS", "1") != "0"

        if "documents" in raw_data:
            for doc_id, doc_data in raw_data["documents"].items():
                try:
                    # Extract content from various Granola fields
                    content_parts = []

                    # Try notes_plain first (cleanest format)
                    if doc_data.get("notes_plain"):
                        content_parts.append(doc_data["notes_plain"])

                    # Try notes_markdown as backup
                    elif doc_data.get("notes_markdown"):
                        content_parts.append(doc_data["notes_markdown"])

                    # Try to extract from structured notes field
                    elif doc_data.get("notes") and isinstance(doc_data["notes"], dict):
                        notes_content = self._extract_structured_notes(
                            doc_data["notes"]
                        )
                        if notes_content:
                            content_parts.append(notes_content)

                    # Fallback to document panels when traditional fields are empty
                    if parse_panels and not any(
                        isinstance(part, str) and part.strip() for part in content_parts
                    ):
                        panel_text = self._extract_document_panel_content(
                            document_panels.get(doc_id)
                        )
                        if panel_text:
                            content_parts.append(panel_text)

                    # Add overview if available
                    if doc_data.get("overview"):
                        content_parts.append(f"Overview: {doc_data['overview']}")

                    # Add summary if available
                    if doc_data.get("summary"):
                        content_parts.append(f"Summary: {doc_data['summary']}")

                    content = "\n\n".join(content_parts)

                    # Only create document if we have a meeting for it
                    if doc_id in cache_data.meetings:
                        meeting = cache_data.meetings[doc_id]
                        document = MeetingDocument(
                            id=doc_id,
                            meeting_id=doc_id,
                            title=meeting.title or "Untitled Meeting",
                            content=content,
                            document_type="meeting_notes",
                            created_at=meeting.date,
                            tags=[],
                        )
                        cache_data.documents[doc_id] = document

                except Exception as e:
                    print(f"Error extracting document content for {doc_id}: {e}")

        cache_data.last_updated = datetime.now(zoneinfo.ZoneInfo("UTC"))
        return cache_data

    def _extract_structured_notes(self, notes_data: Dict[str, Any]) -> str:
        """Extract text content from Granola's structured notes format."""
        try:
            if not isinstance(notes_data, dict) or "content" not in notes_data:
                return ""

            def extract_text_from_content(content_list):
                text_parts = []
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict):
                            # Handle different content types
                            if item.get("type") == "paragraph" and "content" in item:
                                text_parts.append(
                                    extract_text_from_content(item["content"])
                                )
                            elif item.get("type") == "text" and "text" in item:
                                text_parts.append(item["text"])
                            elif "content" in item:
                                text_parts.append(
                                    extract_text_from_content(item["content"])
                                )
                return " ".join(text_parts)

            return extract_text_from_content(notes_data["content"])

        except Exception as e:
            print(f"Error extracting structured notes: {e}")
            return ""

    def _extract_document_panel_content(self, panel_data: Any) -> str:
        """Extract text content from Granola's documentPanels structure."""
        if not panel_data:
            return ""

        text_parts = []

        def extract_from_node(node: Any):
            if isinstance(node, dict):
                node_type = node.get("type")

                if node_type == "text" and node.get("text"):
                    text_parts.append(node["text"])
                elif "content" in node:
                    extract_from_node(node["content"])
            elif isinstance(node, list):
                for item in node:
                    extract_from_node(item)

        try:
            if isinstance(panel_data, dict):
                # Panels keyed by UUID -> {content: [...]} structure
                for panel_id in sorted(panel_data.keys()):
                    panel = panel_data.get(panel_id)
                    if isinstance(panel, dict):
                        extract_from_node(panel.get("content"))
            elif isinstance(panel_data, list):
                for panel in panel_data:
                    extract_from_node(panel)

        except Exception as exc:
            print(f"Error extracting panel content: {exc}")

        combined = "\n\n".join(
            part.strip()
            for part in text_parts
            if isinstance(part, str) and part.strip()
        )
        return combined.strip()

    def _parse_date_string(self, date_str: Optional[str]) -> Optional[date]:
        """Parse a date string into a date object. Handles 'today', 'yesterday', and YYYY-MM-DD format."""
        if not date_str:
            return None

        date_str = date_str.strip().lower()
        today = datetime.now(self.local_timezone).date()

        if date_str == "today":
            return today
        elif date_str == "yesterday":
            return today - timedelta(days=1)
        else:
            try:
                return datetime.strptime(date_str.upper(), "%Y-%m-%d").date()
            except ValueError:
                return None

    async def _list_meetings(
        self,
        date_str: Optional[str] = None,
        start_date_str: Optional[str] = None,
        end_date_str: Optional[str] = None,
    ) -> List[TextContent]:
        """List meetings for a specific date or date range."""
        if not self.cache_data:
            return [TextContent(type="text", text="No meeting data available")]

        # Determine the date range to filter
        today = datetime.now(self.local_timezone).date()

        if start_date_str and end_date_str:
            # Date range query
            start_date = self._parse_date_string(start_date_str)
            end_date = self._parse_date_string(end_date_str)
            if not start_date or not end_date:
                return [
                    TextContent(
                        type="text",
                        text="Invalid date format. Use YYYY-MM-DD, 'today', or 'yesterday'.",
                    )
                ]
            date_description = f"{start_date} to {end_date}"
        elif date_str:
            # Single date query
            target_date = self._parse_date_string(date_str)
            if not target_date:
                return [
                    TextContent(
                        type="text",
                        text=f"Invalid date format: '{date_str}'. Use YYYY-MM-DD, 'today', or 'yesterday'.",
                    )
                ]
            start_date = target_date
            end_date = target_date
            if date_str.lower() == "today":
                date_description = f"today ({target_date})"
            elif date_str.lower() == "yesterday":
                date_description = f"yesterday ({target_date})"
            else:
                date_description = str(target_date)
        else:
            # Default to today
            start_date = today
            end_date = today
            date_description = f"today ({today})"

        # Filter meetings by date range (using local time)
        matching_meetings = []
        for meeting_id, meeting in self.cache_data.meetings.items():
            local_dt = self._convert_to_local_time(meeting.date)
            meeting_date = local_dt.date()

            if start_date <= meeting_date <= end_date:
                matching_meetings.append((local_dt, meeting))

        # Sort by time
        matching_meetings.sort(key=lambda x: x[0])

        if not matching_meetings:
            return [
                TextContent(
                    type="text", text=f"No meetings found for {date_description}."
                )
            ]

        # Build output
        output_lines = [f"# Meetings for {date_description}\n"]
        output_lines.append(f"Found {len(matching_meetings)} meeting(s):\n")

        current_date = None
        for local_dt, meeting in matching_meetings:
            meeting_date = local_dt.date()

            # Add date header if showing multiple days
            if start_date != end_date and meeting_date != current_date:
                current_date = meeting_date
                output_lines.append(f"\n## {meeting_date.strftime('%A, %B %d, %Y')}\n")

            output_lines.append(f"### {meeting.title or 'Untitled Meeting'}")
            output_lines.append(f"- **Time:** {local_dt.strftime('%H:%M')}")
            output_lines.append(f"- **ID:** {meeting.id}")

            if meeting.participants:
                output_lines.append(
                    f"- **Participants:** {', '.join(meeting.participants)}"
                )
            else:
                output_lines.append("- **Participants:** (none recorded)")

            output_lines.append("")

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _search_meetings(self, query: str, limit: int = 10) -> List[TextContent]:
        """Search meetings by query."""
        if not self.cache_data:
            return [TextContent(type="text", text="No meeting data available")]

        query_lower = query.lower()
        results = []

        for meeting_id, meeting in self.cache_data.meetings.items():
            score = 0

            # Search in title
            if meeting.title and query_lower in meeting.title.lower():
                score += 2

            # Search in participants
            for participant in meeting.participants:
                if query_lower in participant.lower():
                    score += 1

            # Search in transcript content if available
            if meeting_id in self.cache_data.transcripts:
                transcript = self.cache_data.transcripts[meeting_id]
                if query_lower in transcript.content.lower():
                    score += 1

            if score > 0:
                results.append((score, meeting))

        # Sort by relevance and limit results
        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:limit]

        if not results:
            return [
                TextContent(type="text", text=f"No meetings found matching '{query}'")
            ]

        output_lines = [f"Found {len(results)} meeting(s) matching '{query}':\n"]

        for score, meeting in results:
            output_lines.append(
                f"• **{meeting.title or 'Untitled Meeting'}** ({meeting.id})"
            )
            output_lines.append(f"  Date: {self._format_local_time(meeting.date)}")
            if meeting.participants:
                output_lines.append(
                    f"  Participants: {', '.join(meeting.participants)}"
                )
            output_lines.append("")

        return [TextContent(type="text", text="\n".join(output_lines))]

    async def _get_meeting_details(self, meeting_id: str) -> List[TextContent]:
        """Get detailed meeting information."""
        if not self.cache_data or meeting_id not in self.cache_data.meetings:
            return [TextContent(type="text", text=f"Meeting '{meeting_id}' not found")]

        meeting = self.cache_data.meetings[meeting_id]

        details = [
            f"# Meeting Details: {meeting.title or 'Untitled Meeting'}\n",
            f"**ID:** {meeting.id}",
            f"**Date:** {self._format_local_time(meeting.date)}",
        ]

        if meeting.duration:
            details.append(f"**Duration:** {meeting.duration} minutes")

        if meeting.participants:
            details.append(f"**Participants:** {', '.join(meeting.participants)}")

        if meeting.meeting_type:
            details.append(f"**Type:** {meeting.meeting_type}")

        if meeting.platform:
            details.append(f"**Platform:** {meeting.platform}")

        # Add document count
        doc_count = sum(
            1
            for doc in self.cache_data.documents.values()
            if doc.meeting_id == meeting_id
        )
        if doc_count > 0:
            details.append(f"**Documents:** {doc_count}")

        # Add transcript availability
        if meeting_id in self.cache_data.transcripts:
            details.append("**Transcript:** Available")

        return [TextContent(type="text", text="\n".join(details))]

    async def _get_meeting_transcript(self, meeting_id: str) -> List[TextContent]:
        """Get meeting transcript."""
        if not self.cache_data:
            return [TextContent(type="text", text="No meeting data available")]

        if meeting_id not in self.cache_data.transcripts:
            return [
                TextContent(
                    type="text",
                    text=f"No transcript available for meeting '{meeting_id}'",
                )
            ]

        transcript = self.cache_data.transcripts[meeting_id]
        meeting = self.cache_data.meetings.get(meeting_id)

        output = [
            f"# Transcript: {meeting.title if meeting and meeting.title else meeting_id}\n"
        ]

        if transcript.speakers:
            output.append(f"**Speakers:** {', '.join(transcript.speakers)}")

        if transcript.language:
            output.append(f"**Language:** {transcript.language}")

        if transcript.confidence:
            output.append(f"**Confidence:** {transcript.confidence:.2%}")

        output.append("\n## Transcript Content\n")
        output.append(transcript.content)

        return [TextContent(type="text", text="\n".join(output))]

    async def _get_meeting_documents(self, meeting_id: str) -> List[TextContent]:
        """Get meeting documents."""
        if not self.cache_data:
            return [TextContent(type="text", text="No meeting data available")]

        documents = [
            doc
            for doc in self.cache_data.documents.values()
            if doc.meeting_id == meeting_id
        ]

        if not documents:
            return [
                TextContent(
                    type="text", text=f"No documents found for meeting '{meeting_id}'"
                )
            ]

        meeting = self.cache_data.meetings.get(meeting_id)
        output = [
            f"# Documents: {meeting.title if meeting and meeting.title else meeting_id}\n"
        ]
        output.append(f"Found {len(documents)} document(s):\n")

        for doc in documents:
            output.append(f"## {doc.title}")
            output.append(f"**Type:** {doc.document_type}")
            output.append(f"**Created:** {self._format_local_time(doc.created_at)}")

            if doc.tags:
                output.append(f"**Tags:** {', '.join(doc.tags)}")

            output.append(f"\n{doc.content}\n")
            output.append("---\n")

        return [TextContent(type="text", text="\n".join(output))]

    async def _analyze_meeting_patterns(
        self, pattern_type: str, date_range: Optional[Dict] = None
    ) -> List[TextContent]:
        """Analyze patterns across meetings."""
        if not self.cache_data:
            return [TextContent(type="text", text="No meeting data available")]

        meetings = list(self.cache_data.meetings.values())

        # Filter by date range if provided
        if date_range:
            start_date_str = date_range.get("start_date", "1900-01-01")
            end_date_str = date_range.get("end_date", "2100-01-01")

            # Parse dates and ensure timezone-aware
            naive_start = datetime.fromisoformat(start_date_str)
            naive_end = datetime.fromisoformat(end_date_str)

            # Localize naive datetimes to UTC
            if naive_start.tzinfo is None:
                start_date = naive_start.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            else:
                start_date = naive_start

            if naive_end.tzinfo is None:
                end_date = naive_end.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            else:
                end_date = naive_end

            meetings = [m for m in meetings if start_date <= m.date <= end_date]

        if pattern_type == "participants":
            return await self._analyze_participant_patterns(meetings)
        elif pattern_type == "frequency":
            return await self._analyze_frequency_patterns(meetings)
        elif pattern_type == "topics":
            return await self._analyze_topic_patterns(meetings)
        else:
            return [
                TextContent(type="text", text=f"Unknown pattern type: {pattern_type}")
            ]

    async def _analyze_participant_patterns(
        self, meetings: List[MeetingMetadata]
    ) -> List[TextContent]:
        """Analyze participant patterns."""
        participant_counts = {}

        for meeting in meetings:
            for participant in meeting.participants:
                participant_counts[participant] = (
                    participant_counts.get(participant, 0) + 1
                )

        if not participant_counts:
            return [TextContent(type="text", text="No participant data found")]

        sorted_participants = sorted(
            participant_counts.items(), key=lambda x: x[1], reverse=True
        )

        output = [
            f"# Participant Analysis ({len(meetings)} meetings)\n",
            "## Most Active Participants\n",
        ]

        for participant, count in sorted_participants[:10]:
            output.append(f"• **{participant}:** {count} meetings")

        return [TextContent(type="text", text="\n".join(output))]

    async def _analyze_frequency_patterns(
        self, meetings: List[MeetingMetadata]
    ) -> List[TextContent]:
        """Analyze meeting frequency patterns."""
        if not meetings:
            return [TextContent(type="text", text="No meetings found for analysis")]

        # Group by month
        monthly_counts = {}
        for meeting in meetings:
            month_key = meeting.date.strftime("%Y-%m")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1

        output = [
            f"# Meeting Frequency Analysis ({len(meetings)} meetings)\n",
            "## Meetings by Month\n",
        ]

        for month, count in sorted(monthly_counts.items()):
            output.append(f"• **{month}:** {count} meetings")

        avg_per_month = len(meetings) / len(monthly_counts) if monthly_counts else 0
        output.append(f"\n**Average per month:** {avg_per_month:.1f}")

        return [TextContent(type="text", text="\n".join(output))]

    async def _analyze_topic_patterns(
        self, meetings: List[MeetingMetadata]
    ) -> List[TextContent]:
        """Analyze topic patterns from meeting titles."""
        if not meetings:
            return [TextContent(type="text", text="No meetings found for analysis")]

        # Simple keyword extraction from titles
        word_counts = {}
        for meeting in meetings:
            title = meeting.title or ""
            words = title.lower().split()
            for word in words:
                # Filter out common words
                if len(word) > 3 and word not in ["meeting", "call", "sync", "with"]:
                    word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return [
                TextContent(
                    type="text", text="No significant topics found in meeting titles"
                )
            ]

        sorted_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        output = [
            f"# Topic Analysis ({len(meetings)} meetings)\n",
            "## Most Common Topics (from titles)\n",
        ]

        for topic, count in sorted_topics[:15]:
            output.append(f"• **{topic}:** {count} mentions")

        return [TextContent(type="text", text="\n".join(output))]

    def run(self, transport_type: str = "stdio"):
        """Run the server."""
        import asyncio
        from mcp.server.stdio import stdio_server
        from mcp.types import ServerCapabilities

        if transport_type == "stdio":

            async def main():
                # Set up server capabilities for tool support
                capabilities = ServerCapabilities(
                    tools=None  # Indicates tool support is available
                )

                options = InitializationOptions(
                    server_name="granola-mcp-server",
                    server_version="0.1.0",
                    capabilities=capabilities,
                )

                async with stdio_server() as (read_stream, write_stream):
                    await self.server.run(read_stream, write_stream, options)

            return asyncio.run(main())
        else:
            raise ValueError(
                f"Unsupported transport type: {transport_type}. Only 'stdio' is supported."
            )


def main():
    """Main entry point for the server."""
    server = GranolaMCPServer()
    server.run()
