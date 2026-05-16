import { useState } from "react";
import axios from "axios";

import "./App.css";

function App() {

    const [question, setQuestion] = useState("");

    const [answer, setAnswer] = useState("");

    const askQuestion = async () => {

        try {

            const response = await axios.post(
                "http://127.0.0.1:8000/ask",
                {
                    question: question
                }
            );

            setAnswer(
                response.data.answer
            );

        } catch (error) {

            console.log(error);

            setAnswer(
                "Error connecting to backend"
            );
        }
    };

    return (

        <div className="container">

            <h1 className="title">
                PharmaRAG AI Assistant
            </h1>

            <textarea
                className="input-box"
                placeholder="Ask pharmaceutical question..."
                value={question}
                onChange={(e) =>
                    setQuestion(e.target.value)
                }
            />

            <button
                className="ask-btn"
                onClick={askQuestion}
            >
                Ask AI
            </button>

            <div className="answer-box">
                {answer}
            </div>

        </div>
    );
}

export default App;