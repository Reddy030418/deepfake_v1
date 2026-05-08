import axios from "axios";

const preferredBase = import.meta.env.VITE_API_BASE_URL?.trim();
const CANDIDATE_BASE_URLS = Array.from(
  new Set(
    [
      preferredBase,
      "http://127.0.0.1:8000",
      "http://localhost:8000",
      "http://127.0.0.1:50997",
      "http://localhost:50997"
    ].filter(Boolean)
  )
);

const initialIndex = (() => {
  const raw = Number(localStorage.getItem("ds_api_index") || 0);
  if (!Number.isFinite(raw) || raw < 0 || raw >= CANDIDATE_BASE_URLS.length) {
    return 0;
  }
  return raw;
})();

let activeBaseIndex = initialIndex;

function setActiveBaseIndex(nextIndex) {
  activeBaseIndex = Math.max(0, Math.min(nextIndex, CANDIDATE_BASE_URLS.length - 1));
  localStorage.setItem("ds_api_index", String(activeBaseIndex));
  client.defaults.baseURL = CANDIDATE_BASE_URLS[activeBaseIndex];
}

const client = axios.create({
  baseURL: CANDIDATE_BASE_URLS[activeBaseIndex] || "http://127.0.0.1:8000",
  timeout: 30000
});

client.interceptors.request.use((config) => {
  config.baseURL = CANDIDATE_BASE_URLS[activeBaseIndex] || config.baseURL;
  const token = localStorage.getItem("ds_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error?.config;
    const isNetworkError = error?.code === "ERR_NETWORK";
    const isRelativeUrl = config?.url && !/^https?:\/\//i.test(config.url);

    if (isNetworkError && config && isRelativeUrl) {
      const attempted = Number(config.__apiFallbackAttempt || 0);
      if (attempted < CANDIDATE_BASE_URLS.length - 1) {
        const nextIndex = (activeBaseIndex + 1) % CANDIDATE_BASE_URLS.length;
        setActiveBaseIndex(nextIndex);
        config.__apiFallbackAttempt = attempted + 1;
        config.baseURL = CANDIDATE_BASE_URLS[nextIndex];
        return client.request(config);
      }
    }

    return Promise.reject(error);
  }
);

export default client;
