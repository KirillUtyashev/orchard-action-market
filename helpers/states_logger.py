# --- START OF FILE helpers/datalogger.py ---

import numpy as np
import os


class HtmlDataLogger:
    def __init__(self, filepath, batch_size):
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer = []
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Initialize the HTML file
        with open(self.filepath, "w") as f:
            f.write("<html><head><title>Training Data Log</title>")
            f.write("<style> table { border-collapse: collapse; margin: 25px; }")
            f.write(
                "td, th { border: 1px solid #dddddd; text-align: center; padding: 8px; font-family: monospace; }"
            )
            f.write("</style></head><body>")
            f.write("<h1>Training Data Log</h1>")

    def log_experience(self, old_state, new_state, reward):
        """Adds a single experience to the buffer."""
        self.buffer.append(
            {"old_state": old_state, "new_state": new_state, "reward": reward}
        )
        # When the buffer is full, write it to the file
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def _format_state_to_html(self, state_tensor):
        """Converts a 3D state tensor into an HTML table for visualization."""
        html = ""
        # Channel 0: Apples
        html += "<p>Apples:</p><table>"
        for row in state_tensor[0]:
            html += "<tr>" + "".join(f"<td>{int(cell)}</td>" for cell in row) + "</tr>"
        html += "</table>"

        # Channel 1: Agents
        html += "<p>Agents:</p><table>"
        for row in state_tensor[1]:
            html += "<tr>" + "".join(f"<td>{int(cell)}</td>" for cell in row) + "</tr>"
        html += "</table>"
        return html

    def flush(self):
        """Writes the buffered experiences to the HTML file."""
        if not self.buffer:
            return

        with open(self.filepath, "a") as f:
            batch_num = os.path.getsize(self.filepath) // 1000  # A rough batch number
            f.write(
                f"<h2>Batch starting around step {batch_num * self.batch_size}</h2>"
            )
            for i, exp in enumerate(self.buffer):
                f.write(f"<h3>Experience {i+1} in Batch</h3>")
                f.write(
                    "<table><tr><th>Old State</th><th>New State</th><th>Reward</th></tr>"
                )
                f.write("<tr>")
                f.write(f'<td>{self._format_state_to_html(exp["old_state"])}</td>')
                f.write(f'<td>{self._format_state_to_html(exp["new_state"])}</td>')
                f.write(f'<td><h2>{exp["reward"]:.2f}</h2></td>')
                f.write("</tr></table><hr>")

        self.buffer.clear()

    def close(self):
        """Writes any remaining data and closes the HTML tags."""
        self.flush()
        with open(self.filepath, "a") as f:
            f.write("</body></html>")
