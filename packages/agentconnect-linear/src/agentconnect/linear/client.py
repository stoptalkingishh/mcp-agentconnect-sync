"""Thin Linear GraphQL client (spec §14).

The transport is injectable: `transport(query, variables) -> data`. Tests pass a
fake and never touch the network; production passes nothing and gets httpx.
Linear is a *mirror*, so every call here is a write to somebody else's database —
nothing the backplane needs to be correct depends on the response.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

LINEAR_ENDPOINT = "https://api.linear.app/graphql"

#: (query, variables) -> the GraphQL ``data`` object.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class LinearError(RuntimeError):
    pass


_CREATE_ISSUE = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier url title }
  }
}
"""

_UPDATE_ISSUE = """
mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue { id identifier url title }
  }
}
"""

_CREATE_COMMENT = """
mutation CommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) { success comment { id url } }
}
"""

_LIST_LABELS = """
query Labels($teamId: String!) {
  team(id: $teamId) { labels(first: 100) { nodes { id name } } }
}
"""

_GET_ISSUE = """
query Issue($id: String!) {
  issue(id: $id) { id identifier url title description state { name } }
}
"""


def _httpx_transport(api_key: str, endpoint: str) -> Transport:
    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        import httpx  # lazy: the [linear] extra, not a core dependency

        response = httpx.post(
            endpoint,
            json={"query": query, "variables": variables},
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=30.0,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise LinearError(f"Linear API error: {body['errors']}")
        return body.get("data") or {}

    return transport


class LinearClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        endpoint: str = LINEAR_ENDPOINT,
    ) -> None:
        if transport is None:
            key = api_key or os.environ.get("LINEAR_API_KEY")
            if not key:
                raise LinearError(
                    "no Linear transport and no LINEAR_API_KEY; refusing to guess credentials"
                )
            transport = _httpx_transport(key, endpoint)
        self._transport = transport

    def _call(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        return self._transport(query, variables) or {}

    def create_issue(
        self, team_id: str, title: str, description: str = "",
        label_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"teamId": team_id, "title": title, "description": description}
        if label_ids:
            payload["labelIds"] = label_ids
        data = self._call(_CREATE_ISSUE, {"input": payload})
        result = data.get("issueCreate") or {}
        if not result.get("success"):
            raise LinearError(f"issueCreate failed: {data}")
        return result["issue"]

    def update_issue(
        self, issue_id: str, title: Optional[str] = None, description: Optional[str] = None,
        label_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        if label_ids is not None:
            payload["labelIds"] = label_ids
        data = self._call(_UPDATE_ISSUE, {"id": issue_id, "input": payload})
        result = data.get("issueUpdate") or {}
        if not result.get("success"):
            raise LinearError(f"issueUpdate failed: {data}")
        return result["issue"]

    def create_comment(self, issue_id: str, body: str) -> dict[str, Any]:
        data = self._call(_CREATE_COMMENT, {"input": {"issueId": issue_id, "body": body}})
        result = data.get("commentCreate") or {}
        if not result.get("success"):
            raise LinearError(f"commentCreate failed: {data}")
        return result.get("comment") or {}

    def list_labels(self, team_id: str) -> dict[str, str]:
        """``{label_name: label_id}``. Labels that do not exist are skipped by the
        sync rather than created — the backplane does not administer the tracker."""
        data = self._call(_LIST_LABELS, {"teamId": team_id})
        nodes = (((data.get("team") or {}).get("labels") or {}).get("nodes")) or []
        return {n["name"]: n["id"] for n in nodes}

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        return (self._call(_GET_ISSUE, {"id": issue_id}) or {}).get("issue") or {}
