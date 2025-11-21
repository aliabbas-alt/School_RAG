const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

app.post("/chat", async (req, res) => {
  const { query } = req.body;

  try {
    // Forward query to Python FastAPI service
    const pyRes = await axios.post("http://192.168.0.3:5000/agent", { query });

    // Send Python agent response back to frontend
    res.json(pyRes.data);
  } catch (err) {
    console.error("Error calling Python agent:", err.message);
    res.status(502).json({ error: "Agent service unavailable" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Node backend running on http://localhost:${PORT}`);
});