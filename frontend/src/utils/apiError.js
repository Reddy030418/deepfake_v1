export function getErrorMessage(error, fallback = "Something went wrong") {
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
