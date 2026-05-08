export function getAdminToken() {
  return localStorage.getItem("ds_admin_token");
}

export function setAdminToken(token) {
  localStorage.setItem("ds_admin_token", token);
}

export function clearAdminToken() {
  localStorage.removeItem("ds_admin_token");
}

export function isAdminAuthenticated() {
  return Boolean(getAdminToken());
}
