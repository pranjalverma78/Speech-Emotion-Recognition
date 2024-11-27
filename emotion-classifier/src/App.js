import React, { useState } from 'react';
import AudioUpload from './components/AudioUpload';
import WaveformDisplay from './components/WaveformDisplay';
import './App.css';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [emotion, setEmotion] = useState('');

  const predictEmotion = async () => {
    if (!audioFile) return;

    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction failed');

      const result = await response.json();
      setEmotion(result.emotion); // Update state with the prediction result
      console.log(result);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div
      className="container-fluid d-flex align-items-center justify-content-center vh-100 text-center"
      style={{
        backgroundImage: `url('https://wallpapers.com/images/hd/pop-music-kws85adurytn2fe5.jpg')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        backgroundColor: 'rgba(0, 0, 0, 0.5)', // Fallback color
        backgroundBlendMode: 'overlay',
      }}
    >
      <div
        className="p-5 rounded shadow"
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.8)', // Semi-transparent background
          maxWidth: '600px',
        }}
      >
        <h1 className="display-4 fw-bold mb-4">Discover Your Emotion</h1>
        <p 
          className="text-muted mb-4 fs-5 fw-light" 
          style={{
            fontStyle: 'italic',
            textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
            letterSpacing: '0.5px',
          }}
        >
          Upload an audio file and let us analyze your emotions!
        </p>

        <AudioUpload setAudioFile={setAudioFile} />

        {audioFile && (
          <div className="mt-4">
            <WaveformDisplay audioFile={audioFile} />
            <button className="btn btn-primary mt-3 px-4 py-2" onClick={predictEmotion}>
              Predict Emotion
            </button>
          </div>
        )}

        {emotion && (
          <div className="mt-4">
            <p className="fs-5 text-success">
            <strong 
              style={{
                fontSize: '2rem',
                fontWeight: 'bold',
                color: '#17032b',
                backgroundColor: 'white',
                padding: '0.5rem 1rem', // Increased padding for wider background
                borderRadius: '50px', // Fully rounded background
                textShadow: '0px 2px 4px rgba(0, 0, 0, 0.2)', // Subtle shadow for depth
                display: 'inline-block', // Ensures the background only covers the text
                width: 'auto', // Example color, use any you'd like
              }}
            >
              {emotion}
            </strong>

            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
