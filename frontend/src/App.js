import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [confidence, setConfidence] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setCitations([]);
    setConfidence("");

    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        question,
      });
      
      setAnswer(res.data.answer || "No answer received.");
      setCitations(res.data.citations || []);
      setConfidence(res.data.confidence || "unknown");
      
    } catch (error) {
      setAnswer("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion("");
    setAnswer("");
    setCitations([]);
    setConfidence("");
  };

  const handleCopyAnswer = async () => {
    if (answer) {
      try {
        await navigator.clipboard.writeText(answer);
        alert("Answer copied to clipboard!");
      } catch (err) {
        console.error("Failed to copy: ", err);
      }
    }
  };

  return (
    <div className="chat-container">
      <h1>AI HR Chatbot 💬</h1>
      <div className="chat-box">
        <textarea
          placeholder="Ask any HR-related question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="button-group">
          <button onClick={handleAsk} disabled={loading}>
            {loading ? "Thinking..." : "Ask"}
          </button>
          <button onClick={handleClear} className="clear-btn">
            Clear
          </button>
          {answer && (
            <button onClick={handleCopyAnswer} className="copy-btn">
              Copy Answer
            </button>
          )}
        </div>
      </div>

      {answer && (
        <div className="response">
          <h3>Answer:</h3>
          <p>{answer}</p>

          {/* Show Confidence Level */}
          {confidence && (
            <div className="confidence">
              <strong>Confidence:</strong> {confidence}
              {confidence === "low" && (
                <div className="hr-contact">
                  🤔 Can't find a clear answer? <a href="mailto:hr@company.com">Contact HR</a>
                </div>
              )}
            </div>
          )}

          {/* Show Citations */}
          {citations && citations.length > 0 && (
            <div className="citations">
              <h4>📚 Sources:</h4>
              {citations.map((citation, index) => (
                <div key={index} className="citation">
                  <div><strong>Document:</strong> {citation.doc_id}</div>
                  <div><strong>Text:</strong> {citation.snippet}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;