import React from 'react';
// import './AudioUpload.css';

function AudioUpload({ setAudioFile }) {
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
    }
  };

  return (
    <div className="file-converter-upload">
      <div 
        className="file-converter-upload__title text-uppercase fw-bold mb-3" 
        style={{
          fontSize: '1.5rem',
          color: '#4A90E2',
          textShadow: '0px 2px 4px rgba(0, 0, 0, 0.2)',
          letterSpacing: '1px',
          paddingBottom: '0.5rem',
          display: 'inline-block',
        }}
      >
        Upload file to convert
      </div>

      <div className="file-converter-upload__area">
        <label className="app-button" htmlFor="audio-upload-input">
          <span className="app-button__text">Choose file</span>
        </label>
        <input 
          id="audio-upload-input"
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          className="file-converter-upload__input"
          hidden
        />
        <div className="file-converter-upload__text" style={{
          paddingTop: '1rem',
          color: '#3c4c75'
        }}>Drag and drop your audio here to add</div>
      </div>
    </div>
  );
}

export default AudioUpload;
