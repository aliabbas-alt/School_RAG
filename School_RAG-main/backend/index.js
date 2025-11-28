const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(bodyParser.json());
app.use(cors());

app.post("/query", (req, res) => {
  const { query, schoolId } = req.body;

  let data = "";

  // Use Python -c to call run_agent_api directly
  const py = spawn("python", [
    "-c",
    `
import json
from run_agent import run_agent_api
print(json.dumps({"answer": run_agent_api("${query}", "${schoolId || 'demo_school'}")}))
    `,
  ]);

  py.stdout.on("data", (chunk) => {
    data += chunk.toString();
  });

  py.stderr.on("data", (err) => {
    console.error("Python error:", err.toString());
  });

  py.on("close", () => {
    try {
      const parsed = JSON.parse(data.trim());
      res.json(parsed);
    } catch {
      res.json({ answer: data.trim() });
    }
  });
});

app.listen(3000, () => {
  console.log("🚀 API running on http://localhost:3000");
});