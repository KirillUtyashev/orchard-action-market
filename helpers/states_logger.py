# File: helpers/states_logger.py

import os
import numpy as np
from typing import Dict


class HtmlDataLogger:
    def __init__(self, filepath: str, max_entries: int = 20):
        self.filepath = filepath
        self.max_entries = max_entries
        self.entries_logged = 0

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            f.write("<html><head><title>Evaluation Data Log</title>")
            f.write(
                "<style> table, th, td { border: 1px solid black; border-collapse: collapse; margin: 15px; font-family: monospace; text-align: center; padding: 5px; } </style>"
            )
            f.write("</head><body><h1>Evaluation Log</h1>")

    def _format_raw_state_to_html(self, state: dict) -> str:
        """Formats the raw agent/apple grids into a simple HTML table."""
        agents = state["agents"]
        apples = state["apples"]
        html = "<table><tr><th>Agents</th><th>Apples</th></tr><tr>"

        # Agents grid
        html += "<td><table>"
        for row in agents:
            html += "<tr>" + "".join(f"<td>{int(cell)}</td>" for cell in row) + "</tr>"
        html += "</table></td>"

        # Apples grid
        html += "<td><table>"
        for row in apples:
            html += "<tr>" + "".join(f"<td>{int(cell)}</td>" for cell in row) + "</tr>"
        html += "</table></td>"

        html += "</tr></table>"
        return html

    def _format_processed_state_to_html(self, state_tensor: np.ndarray) -> str:
        """Formats a multi-channel processed state tensor into HTML tables."""
        num_channels = state_tensor.shape[0]
        channel_names = ["Apples", "Other Agents", "Self Agent"]
        if num_channels == 2:  # Centralized case
            channel_names = ["Apples", "All Agents"]

        html = "<table>"
        for i in range(num_channels):
            html += f"<tr><th>Channel {i}: {channel_names[i]}</th></tr>"
            html += "<tr><td><table>"
            for row in state_tensor[i]:
                html += (
                    "<tr>" + "".join(f"<td>{cell:.1f}</td>" for cell in row) + "</tr>"
                )
            html += "</table></td></tr>"
        html += "</table>"
        return html

    def log_experience(
        self,
        acting_agent_id: int,
        old_raw_state: dict,
        new_raw_state: dict,
        reward_vector: np.ndarray,
        processed_views: Dict[int, Dict[str, np.ndarray]],
    ):
        """
        Logs a complete experience, including raw states and all per-agent processed views.

        Args:
            acting_agent_id: The ID of the agent who took the action.
            old_raw_state: The raw environment state before the action.
            new_raw_state: The raw environment state after the action.
            reward_vector: The reward received by each agent.
            processed_views: A dictionary mapping agent_id to its processed views.
                             e.g., {0: {'old': processed_old_0, 'new': processed_new_0}, 1: ...}
        """
        if self.entries_logged >= self.max_entries:
            return

        with open(self.filepath, "a") as f:
            f.write(
                f"<hr><h3>Entry #{self.entries_logged + 1} (Agent {acting_agent_id} acted)</h3>"
            )

            # --- Column for Old State ---
            f.write("<table><tr valign='top'><td>")
            f.write("<h4>Old State (Raw)</h4>")
            f.write(self._format_raw_state_to_html(old_raw_state))
            f.write("</td>")

            # --- Column for New State ---
            f.write("<td>")
            f.write("<h4>New State (Raw)</h4>")
            f.write(self._format_raw_state_to_html(new_raw_state))
            f.write("</td></tr></table>")

            # --- Table for Per-Agent Views and Rewards ---
            f.write("<h4>Per-Agent Processed Views & Rewards</h4>")
            f.write(
                "<table><tr><th>Agent ID</th><th>Processed Old State</th><th>Processed New State</th><th>Reward</th></tr>"
            )

            for agent_id, views in processed_views.items():
                reward = reward_vector[agent_id]
                f.write(f"<tr valign='top'><td>{agent_id}</td>")
                f.write(
                    f"<td>{self._format_processed_state_to_html(views['old'])}</td>"
                )
                f.write(
                    f"<td>{self._format_processed_state_to_html(views['new'])}</td>"
                )
                f.write(f"<td><b>{reward:.2f}</b></td></tr>")

            f.write("</table>")

        self.entries_logged += 1
        if self.entries_logged >= self.max_entries:
            with open(self.filepath, "a") as f:
                f.write("<h3>Max log entries reached.</h3>")

    def close(self):
        """Writes any remaining data and closes the HTML tags."""
        with open(self.filepath, "a") as f:
            f.write("</body></html>")
