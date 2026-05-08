import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import client from "../api/client";
import { getErrorMessage } from "../utils/apiError";

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export default function ImageDetectionPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    async function loadMetrics() {
      try {
        const { data } = await client.get("/model-metrics");
        if (active) {
          setModelMetrics(data);
        }
      } catch {
        if (active) {
          setModelMetrics(null);
        }
      }
    }

    loadMetrics();
    return () => {
      active = false;
    };
  }, []);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError("");
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await client.post("/predict", formData);
      setResult(data);
    } catch (err) {
      setError(getErrorMessage(err, "Image prediction failed"));
    } finally {
      setLoading(false);
    }
  };

  const onFileChange = (e) => {
    const selected = e.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError("");

    if (selected) {
      setPreviewUrl(URL.createObjectURL(selected));
    } else {
      setPreviewUrl("");
    }
  };

  return (
    <main className="page">
      <div className="topbar"><Link to="/app/dashboard">Back</Link></div>
      <section className="card wide">
        <h2>Image Deepfake Detection</h2>

        <form onSubmit={onSubmit} className="form">
          <input type="file" accept="image/*" onChange={onFileChange} />
          <button className="btn" disabled={!file || loading}>
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {previewUrl && (
          <div className="card" style={{ marginTop: "0.8rem" }}>
            <h3>Uploaded Image</h3>
            <img
              src={previewUrl}
              alt="Uploaded preview"
              style={{
                width: "100%",
                maxHeight: "360px",
                objectFit: "contain",
                borderRadius: "0.6rem",
                marginTop: "0.6rem"
              }}
            />
          </div>
        )}

        {result && (
          <div className="card" style={{ marginTop: "0.8rem" }}>
            <h3>Inference Result</h3>
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            <p><strong>Media Type:</strong> {result.media_type || "image"}</p>
            <p><strong>Processing Time:</strong> {result.processing_time_ms ?? "N/A"} ms</p>
          </div>
        )}

        <div className="card" style={{ marginTop: "0.8rem" }}>
          <h3>Model Quality Metrics</h3>
          <p><strong>Accuracy:</strong> {formatMetric(modelMetrics?.accuracy)}</p>
          <p><strong>AUC:</strong> {formatMetric(modelMetrics?.auc)}</p>
          <p><strong>Precision:</strong> {formatMetric(modelMetrics?.precision)}</p>
          <p><strong>Recall:</strong> {formatMetric(modelMetrics?.recall)}</p>
          <p><strong>F1 Score:</strong> {formatMetric(modelMetrics?.f1)}</p>
          <p><strong>Specificity:</strong> {formatMetric(modelMetrics?.specificity)}</p>
          <p><strong>Balanced Accuracy:</strong> {formatMetric(modelMetrics?.balanced_accuracy)}</p>
          <p><strong>Decision Threshold:</strong> {modelMetrics?.threshold ?? "N/A"}</p>
          {modelMetrics?.notes && <p><strong>Notes:</strong> {modelMetrics.notes}</p>}
        </div>
      </section>
    </main>
  );
}
