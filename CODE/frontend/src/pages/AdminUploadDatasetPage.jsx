import { useState } from "react";
import { Link } from "react-router-dom";

export default function AdminUploadDatasetPage() {
  const [file, setFile] = useState(null);

  const onSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    window.alert(`Dataset queued: ${file.name}`);
  };

  return (
    <main className="page">
      <div className="topbar"><Link to="/admin-ui/dashboard">Back</Link></div>
      <section className="card wide">
        <h2>Upload Dataset</h2>
        <form onSubmit={onSubmit} className="form">
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button className="btn" disabled={!file}>Upload</button>
        </form>
      </section>
    </main>
  );
}
