import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';

function WaveformDisplay({ audioFile }) {
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);

  useEffect(() => {
    if (audioFile) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#a864ed',
        progressColor: '#4353FF',
        cursorColor: '#4353FF',
        barWidth: 2,
        barHeight: 1,
        barGap: 3,
        height: 100,
      });

      const reader = new FileReader();
      reader.onload = (e) => {
        wavesurferRef.current.loadBlob(new Blob([e.target.result]));
      };
      reader.readAsArrayBuffer(audioFile);

      return () => {
        wavesurferRef.current.destroy();
      };
    }
  }, [audioFile]);

  return <div ref={waveformRef} className="waveform"></div>;
}

export default WaveformDisplay;
