import React, { useState } from "react";


function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

   const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Détection Pneumonia</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Prédire</button>

      {prediction && (
        <div style={{ marginTop: "20px" }}>
          <h2>Prédiction : {prediction.prediction}</h2>
          <p>Probabilité : {prediction.probability.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}

export default App;