import { useState } from "react";
import { Link } from "react-router-dom";
import client from "../api/client";
import { getErrorMessage } from "../utils/apiError";

export default function VideoDetectionPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
      setError(getErrorMessage(err, "Video prediction failed"));
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
        <h2>Video Deepfake Detection</h2>

        <form onSubmit={onSubmit} className="form">
          <input type="file" accept="video/mp4,video/*" onChange={onFileChange} />
          <button className="btn" disabled={!file || loading}>
            {loading ? "Sampling Frames..." : "Analyze Video"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {previewUrl && (
          <div className="card" style={{ marginTop: "0.8rem" }}>
            <h3>Uploaded Video</h3>
            <video
              src={previewUrl}
              controls
              style={{
                width: "100%",
                maxHeight: "360px",
                borderRadius: "0.6rem",
                marginTop: "0.6rem",
                background: "#000"
              }}
            />
          </div>
        )}

        {result && (
          <div className="card" style={{ marginTop: "0.8rem" }}>
            <h3>Analysis Details</h3>
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            <p><strong>Media Type:</strong> {result.media_type || "video"}</p>
            <p><strong>Processing Time:</strong> {result.processing_time_ms ?? "N/A"} ms</p>
            <p><strong>File Name:</strong> {file?.name || "N/A"}</p>
            <p><strong>File Size:</strong> {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "N/A"}</p>
          </div>
        )}
      </section>
    </main>
  );
}
