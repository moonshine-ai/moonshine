import './App.css';
import { useState } from 'react';

import SpeechInputButton from './SpeechInputButton.js';

function App() {
  const [transcript, setTranscript] = useState("")

  return (
    <div className="App">
      <SpeechInputButton onUpdate={setTranscript} />
      <textarea value={transcript}>
      </textarea>
    </div>
  );
}

export default App;
