import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import client from "../api/client";

const metricItems = [
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "auc", label: "ROC AUC" },
  { key: "specificity", label: "Specificity" },
  { key: "balanced_accuracy", label: "Balanced Accuracy" }
];

const currentAlgorithmRows = [
  {
    stage: "Current Training",
    algorithm: "MobileNetV2 Transfer Learning (CNN)",
    whyUsed: "Fast, stable baseline for image deepfake detection with limited data",
    featureMode: "Automatic feature extraction by deep network",
    output: "Binary classification (authentic vs deepfake)"
  },
  {
    stage: "Loss & Optimizer",
    algorithm: "Binary Crossentropy + Adam",
    whyUsed: "Standard for binary image classification tasks",
    featureMode: "Optimizes model weights end-to-end",
    output: "Probability score (0-1)"
  }
];

const mlVsDlRows = [
  {
    family: "Traditional ML",
    examples: "Linear Regression, Logistic Regression, Decision Tree, Random Forest, XGBoost",
    featureEngineering: "Manual (human-designed features required)",
    strength: "Works well on structured/tabular data, faster on small datasets",
    limitation: "Less effective on raw images/videos without strong feature pipeline"
  },
  {
    family: "Deep Learning (DL)",
    examples: "CNN (MobileNet/ResNet/Xception), ViT/Transformers, 3D CNN for video",
    featureEngineering: "Automatic (model learns features directly from data)",
    strength: "Best for complex image/video patterns such as deepfake artifacts",
    limitation: "Needs more data, compute, and training discipline"
  }
];

const algorithmCatalogRows = [
  {
    family: "ML",
    algorithm: "Linear / Logistic Regression",
    whereItFits: "Baseline on hand-crafted forensic features",
    featureStyle: "Manual feature engineering required",
    expectedResult: "Fast, interpretable, usually lower image accuracy than CNN"
  },
  {
    family: "ML",
    algorithm: "Decision Tree / Random Forest",
    whereItFits: "Simple explainable baseline for tabular features",
    featureStyle: "Manual engineered features",
    expectedResult: "Stable start, moderate performance"
  },
  {
    family: "ML",
    algorithm: "XGBoost / LightGBM",
    whereItFits: "Strong tabular benchmark if features are high quality",
    featureStyle: "Manual + statistical features",
    expectedResult: "Often better than basic ML baselines"
  },
  {
    family: "DL",
    algorithm: "CNN (MobileNetV2)",
    whereItFits: "Current production-friendly image model",
    featureStyle: "Automatic feature extraction",
    expectedResult: "Fast inference with good balance"
  },
  {
    family: "DL",
    algorithm: "CNN (EfficientNetB0 / ResNet50 / Xception)",
    whereItFits: "Accuracy-focused image deepfake detection",
    featureStyle: "Automatic feature extraction",
    expectedResult: "Higher ceiling with more training time"
  },
  {
    family: "DL",
    algorithm: "Transformers (ViT / Swin / CLIP-style encoders)",
    whereItFits: "Large-scale image datasets",
    featureStyle: "Automatic + global-context learning",
    expectedResult: "Can be excellent but compute heavy"
  },
  {
    family: "DL",
    algorithm: "Video Models (3D CNN / TimeSformer)",
    whereItFits: "Full video-level deepfake analysis",
    featureStyle: "Spatial + temporal automatic features",
    expectedResult: "Best video quality, highest hardware cost"
  }
];

const metricExplainRows = [
  { metric: "Accuracy", meaning: "How many total predictions are correct.", easyRead: "Higher is better overall." },
  { metric: "Precision", meaning: "When model says deepfake, how often that is correct.", easyRead: "Higher means fewer false alarms." },
  { metric: "Recall", meaning: "How many actual deepfakes the model catches.", easyRead: "Higher means fewer missed deepfakes." },
  { metric: "F1 Score", meaning: "Balanced score of precision and recall.", easyRead: "Best single quality score for imbalance." },
  { metric: "ROC AUC", meaning: "How well model separates real vs fake across thresholds.", easyRead: "Closer to 1.0 is stronger ranking power." },
  { metric: "Specificity", meaning: "How well authentic media is correctly identified.", easyRead: "Higher means fewer real files flagged as fake." },
  { metric: "Balanced Accuracy", meaning: "Average accuracy across both classes.", easyRead: "Useful when class counts are uneven." }
];

const ROC_FRAME = { x0: 36, y0: 188, w: 230, h: 150 };
const metricHelp = {
  accuracy: "Overall correctness: (TP + TN) / Total.",
  f1: "F1 combines Precision and Recall. Higher means balanced quality.",
  precision: "Of all predicted deepfakes, how many were truly deepfake.",
  recall: "Of all true deepfakes, how many were detected.",
  auc: "ROC AUC measures ranking quality across all thresholds.",
  specificity: "True Negative Rate: how well authentic media is identified.",
  balanced_accuracy: "Average of recall for both classes (robust for imbalance).",
  tn: "True Negative: authentic predicted authentic.",
  fp: "False Positive: authentic predicted deepfake.",
  fn: "False Negative: deepfake predicted authentic.",
  tp: "True Positive: deepfake predicted deepfake.",
  threshold: "Decision cutoff used to convert probability into class label."
};

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "N/A";
  return `${(Number(v) * 100).toFixed(2)}%`;
}

function qualityLabel(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "Unknown";
  if (n >= 0.9) return "Excellent";
  if (n >= 0.85) return "Target Reached";
  if (n >= 0.75) return "Good";
  if (n >= 0.6) return "Needs Improvement";
  return "Poor";
}

export default function ModelDashboardPage() {
  const [metrics, setMetrics] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");
  const [rocHoverIdx, setRocHoverIdx] = useState(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [metricsRes, comparisonRes] = await Promise.all([
          client.get("/model-metrics"),
          client.get("/model-comparison")
        ]);
        if (active) {
          setMetrics(metricsRes.data);
          setComparison(comparisonRes.data);
        }
      } catch (e) {
        if (active) setError(e?.message || "Could not load model metrics");
      }
    })();
    return () => { active = false; };
  }, []);

  const rocPoints = useMemo(() => {
    const points = metrics?.roc_curve_points;
    if (Array.isArray(points) && points.length > 1) {
      return points;
    }
    return [
      { fpr: 0, tpr: 0 },
      { fpr: 1, tpr: 1 }
    ];
  }, [metrics]);

  const rocPlotPoints = useMemo(() => {
    return rocPoints.map((p) => {
      const fpr = Math.max(0, Math.min(1, Number(p.fpr || 0)));
      const tpr = Math.max(0, Math.min(1, Number(p.tpr || 0)));
      return {
        fpr,
        tpr,
        x: ROC_FRAME.x0 + (fpr * ROC_FRAME.w),
        y: ROC_FRAME.y0 - (tpr * ROC_FRAME.h)
      };
    });
  }, [rocPoints]);

  const rocPath = useMemo(() => {
    return rocPlotPoints
      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(" ");
  }, [rocPlotPoints]);

  const onRocMove = (e) => {
    if (!rocPlotPoints.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let bestIdx = 0;
    let bestDist = Number.POSITIVE_INFINITY;

    rocPlotPoints.forEach((p, idx) => {
      const d = ((p.x - mx) ** 2) + ((p.y - my) ** 2);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = idx;
      }
    });

    setRocHoverIdx(bestIdx);
  };

  const cm = metrics?.confusion_matrix;
  const cm2x2 = Array.isArray(cm) && cm.length === 2 && Array.isArray(cm[0]) && Array.isArray(cm[1]);
  const tn = cm2x2 ? Number(cm[0][0] || 0) : 0;
  const fp = cm2x2 ? Number(cm[0][1] || 0) : 0;
  const fn = cm2x2 ? Number(cm[1][0] || 0) : 0;
  const tp = cm2x2 ? Number(cm[1][1] || 0) : 0;
  const cmMax = Math.max(tn, fp, fn, tp, 1);

  const cellStyle = (value, tone = "blue") => {
    const ratio = Math.max(0, Math.min(1, Number(value || 0) / cmMax));
    if (tone === "green") {
      return { background: `rgba(14, 143, 89, ${0.14 + ratio * 0.66})` };
    }
    if (tone === "red") {
      return { background: `rgba(185, 36, 54, ${0.12 + ratio * 0.66})` };
    }
    return { background: `rgba(55, 122, 233, ${0.1 + ratio * 0.62})` };
  };

  const correct = tn + tp;
  const incorrect = fp + fn;
  const total = correct + incorrect;
  const correctDeg = total > 0 ? (correct / total) * 360 : 0;
  const accuracy = Number(metrics?.accuracy);
  const accuracyRatio = Number.isFinite(accuracy) ? Math.max(0, Math.min(1, accuracy)) : 0;

  const f1 = Number(metrics?.f1);
  const f1Ratio = Number.isFinite(f1) ? Math.max(0, Math.min(1, f1)) : 0;
  const f1Deg = f1Ratio * 360;

  const hoveredPoint = rocHoverIdx === null ? null : rocPlotPoints[rocHoverIdx];
  const topModelsRaw = Array.isArray(comparison?.top_models) ? comparison.top_models : [];
  const targetAccuracy = Number(comparison?.target_accuracy ?? 0.9);
  const currentModelName = metrics?.backbone || comparison?.winner_model || "N/A";
  const currentThreshold = metrics?.threshold;
  const currentStatus = qualityLabel(metrics?.accuracy);
  const topModels = useMemo(() => topModelsRaw, [topModelsRaw]);
  const winnerDisplay = comparison?.winner_model || topModels[0]?.model || "N/A";

  return (
    <main className="page">
      <div className="topbar"><Link to="/app/dashboard">Back</Link></div>

      <section className="hero compact">
        <p className="eyebrow">Model Dashboard</p>
        <h1>Performance Analytics</h1>
        <p>Visual view of F1 score, ROC curve, confusion matrix and accuracy.</p>
      </section>

      {error && <p className="error" style={{ marginTop: "0.8rem" }}>{error}</p>}

      <section className="dashboard-grid" style={{ marginTop: "1rem" }}>
        <article className="card">
          <h3 title={metricHelp.accuracy}>Accuracy (Pie)</h3>
          <div className="pie-wrap">
            <div
              className="pie"
              title={metricHelp.accuracy}
              style={{
                background: `conic-gradient(#0e8f59 ${correctDeg}deg, #b92436 ${correctDeg}deg 360deg)`
              }}
            >
              <span>{pct(metrics?.accuracy)}</span>
            </div>
          </div>
          <p title="Correct predictions: TN + TP"><strong>Correct:</strong> {correct}</p>
          <p title="Incorrect predictions: FP + FN"><strong>Incorrect:</strong> {incorrect}</p>
          <p title="Your requested minimum target for production readiness."><strong>Target (90%):</strong> {accuracyRatio >= targetAccuracy ? "Reached" : `Gap ${pct(Math.max(0, targetAccuracy - accuracyRatio))}`}</p>
        </article>

        <article className="card">
          <h3 title={metricHelp.f1}>F1 Score (Gauge)</h3>
          <div className="pie-wrap">
            <div
              className="pie f1-pie"
              title={metricHelp.f1}
              style={{
                background: `conic-gradient(#5a7df6 ${f1Deg}deg, #d8e2ee ${f1Deg}deg 360deg)`
              }}
            >
              <span>{pct(metrics?.f1)}</span>
            </div>
          </div>
          <p title={metricHelp.f1}><strong>Interpretation:</strong> Harmonic balance between precision and recall.</p>
        </article>

        <article className="card">
          <h3>Metric Bars</h3>
          <div className="metric-bars">
            {metricItems.map((m) => {
              const value = Number(metrics?.[m.key]);
              const ratio = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
              return (
                <div key={m.key} className="metric-row" title={metricHelp[m.key]}>
                  <div className="metric-row-head">
                    <span>{m.label}</span>
                    <strong>{pct(metrics?.[m.key])}</strong>
                  </div>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${ratio * 100}%` }} /></div>
                </div>
              );
            })}
          </div>
        </article>

        <article className="card">
          <h3 title={metricHelp.auc}>ROC Curve (Line Chart)</h3>
          <svg
            viewBox="0 0 300 220"
            className="roc-svg"
            role="img"
            aria-label="ROC curve chart"
            onMouseMove={onRocMove}
            onMouseLeave={() => setRocHoverIdx(null)}
          >
            <line x1="36" y1="188" x2="266" y2="188" stroke="#9cb0c5" strokeWidth="1.5" />
            <line x1="36" y1="188" x2="36" y2="38" stroke="#9cb0c5" strokeWidth="1.5" />
            <line x1="36" y1="188" x2="266" y2="38" stroke="#c9d8e8" strokeDasharray="5 5" strokeWidth="1.2" />
            <path d={rocPath} fill="none" stroke="#ff6f3c" strokeWidth="3" strokeLinecap="round" />
            {hoveredPoint && (
              <>
                <circle cx={hoveredPoint.x} cy={hoveredPoint.y} r="4.5" fill="#ff6f3c" />
                <rect
                  x={Math.min(190, Math.max(42, hoveredPoint.x - 14))}
                  y={Math.max(44, hoveredPoint.y - 40)}
                  width="94"
                  height="34"
                  rx="6"
                  fill="rgba(19,33,45,0.92)"
                />
                <text x={Math.min(196, Math.max(48, hoveredPoint.x - 8))} y={Math.max(58, hoveredPoint.y - 26)} className="roc-tip">
                  FPR: {(hoveredPoint.fpr * 100).toFixed(1)}%
                </text>
                <text x={Math.min(196, Math.max(48, hoveredPoint.x - 8))} y={Math.max(72, hoveredPoint.y - 12)} className="roc-tip">
                  TPR: {(hoveredPoint.tpr * 100).toFixed(1)}%
                </text>
              </>
            )}
            <text x="150" y="212" textAnchor="middle" className="roc-label">False Positive Rate</text>
            <text x="12" y="120" transform="rotate(-90 12 120)" textAnchor="middle" className="roc-label">True Positive Rate</text>
          </svg>
          <p title={metricHelp.auc}><strong>ROC AUC:</strong> {pct(metrics?.auc)}</p>
        </article>

        <article className="card">
          <h3>Confusion Matrix</h3>
          <div className="cm-grid">
            <div className="cm-cell cm-head" title={metricHelp.tn}>TN</div>
            <div className="cm-cell cm-head" title={metricHelp.fp}>FP</div>
            <div className="cm-cell cm-value" title={metricHelp.tn} style={cellStyle(tn, "green")}>{tn}</div>
            <div className="cm-cell cm-value" title={metricHelp.fp} style={cellStyle(fp, "red")}>{fp}</div>
            <div className="cm-cell cm-head" title={metricHelp.fn}>FN</div>
            <div className="cm-cell cm-head" title={metricHelp.tp}>TP</div>
            <div className="cm-cell cm-value" title={metricHelp.fn} style={cellStyle(fn, "red")}>{fn}</div>
            <div className="cm-cell cm-value" title={metricHelp.tp} style={cellStyle(tp, "green")}>{tp}</div>
          </div>
          <div className="cm-legend">
            <span><i className="legend-dot ok" /> Correct (TN/TP)</span>
            <span><i className="legend-dot bad" /> Error (FP/FN)</span>
          </div>
          <p title={metricHelp.threshold}><strong>Threshold:</strong> {metrics?.threshold ?? 0.5}</p>
          {metrics?.notes && <p title="Extra context from evaluation output."><strong>Notes:</strong> {metrics.notes}</p>}
        </article>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Algorithm Used In This Project</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Algorithm</th>
                <th>Why Used</th>
                <th>Feature Handling</th>
                <th>Output</th>
              </tr>
            </thead>
            <tbody>
              {currentAlgorithmRows.map((row) => (
                <tr key={row.stage}>
                  <td>{row.stage}</td>
                  <td>{row.algorithm}</td>
                  <td>{row.whyUsed}</td>
                  <td>{row.featureMode}</td>
                  <td>{row.output}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Traditional ML vs Deep Learning (Simple Comparison)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Family</th>
                <th>Common Examples</th>
                <th>Feature Engineering</th>
                <th>Strength</th>
                <th>Limitation</th>
              </tr>
            </thead>
            <tbody>
              {mlVsDlRows.map((row) => (
                <tr key={row.family}>
                  <td>{row.family}</td>
                  <td>{row.examples}</td>
                  <td>{row.featureEngineering}</td>
                  <td>{row.strength}</td>
                  <td>{row.limitation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Algorithms For Better Results (Expanded Catalog)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Family</th>
                <th>Algorithm</th>
                <th>Where It Fits</th>
                <th>Feature Style</th>
                <th>Expected Result</th>
              </tr>
            </thead>
            <tbody>
              {algorithmCatalogRows.map((row) => (
                <tr key={`${row.family}-${row.algorithm}`}>
                  <td>{row.family}</td>
                  <td>{row.algorithm}</td>
                  <td>{row.whereItFits}</td>
                  <td>{row.featureStyle}</td>
                  <td>{row.expectedResult}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Current Model Summary (Easy Read)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1</th>
                <th>ROC AUC</th>
                <th>Threshold</th>
                <th>Status</th>
                <th>Target (90% Accuracy)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{String(currentModelName)}</td>
                <td>{pct(metrics?.accuracy)}</td>
                <td>{pct(metrics?.f1)}</td>
                <td>{pct(metrics?.auc)}</td>
                <td>{currentThreshold !== undefined && currentThreshold !== null ? Number(currentThreshold).toFixed(3) : "N/A"}</td>
                <td>{currentStatus}</td>
                <td>{accuracyRatio >= targetAccuracy ? "Reached" : `Gap: ${pct(Math.max(0, targetAccuracy - accuracyRatio))}`}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Model Comparison (Non-ML Friendly)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1</th>
                <th>ROC AUC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Specificity</th>
                <th>Threshold</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {topModels.map((row) => (
                <tr key={`${row.rank}-${row.model}`}>
                  <td>{row.rank}</td>
                  <td>{row.model}</td>
                  <td>{pct(row.accuracy)}</td>
                  <td>{pct(row.f1)}</td>
                  <td>{pct(row.auc)}</td>
                  <td>{pct(row.precision)}</td>
                  <td>{pct(row.recall)}</td>
                  <td>{pct(row.specificity)}</td>
                  <td>{row.threshold !== undefined && row.threshold !== null ? Number(row.threshold).toFixed(3) : "N/A"}</td>
                  <td>{row.status}</td>
                </tr>
              ))}
              {!topModels.length && (
                <tr>
                  <td colSpan="10">No comparison data yet. Run `train_model_suite.py` first.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>How To Reach 90%+ Accuracy (Action Plan)</h3>
        <p><strong>Winner Model:</strong> {winnerDisplay} | <strong>Status:</strong> {comparison?.winner_status || "Unknown"} | <strong>Target:</strong> {pct(targetAccuracy)}</p>
        <ol>
          {(comparison?.recommendations || []).map((item, index) => (
            <li key={`${index}-${item}`}>{item}</li>
          ))}
        </ol>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h3>Metric Meanings (Simple)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>What It Means</th>
                <th>How To Read</th>
              </tr>
            </thead>
            <tbody>
              {metricExplainRows.map((row) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td>{row.meaning}</td>
                  <td>{row.easyRead}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
