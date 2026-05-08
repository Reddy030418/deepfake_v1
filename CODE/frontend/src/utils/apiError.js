export function getErrorMessage(error, fallback = "Something went wrong") {
  if (error?.code === "ERR_NETWORK") {
    const base = error?.config?.baseURL || "http://127.0.0.1:8000";
    return `Network error: cannot reach backend at ${base}. Start backend and retry.`;
  }

  const detail = error?.response?.data?.detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item?.loc) ? item.loc.join(".") : "field";
        return `${field}: ${item?.msg || "Invalid value"}`;
      })
      .join(" | ");
  }

  if (typeof detail === "string") {
    return detail;
  }

  return error?.response?.data?.error || error?.message || fallback;
}
