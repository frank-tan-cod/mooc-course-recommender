const API_BASE_URL = (process.env.VITE_API_BASE_URL || process.env.API_BASE_URL || "").replace(/\/$/, "");

export default async (request) => {
  if (!API_BASE_URL) {
    return Response.json({ detail: "Missing VITE_API_BASE_URL" }, { status: 500 });
  }

  const incomingUrl = new URL(request.url);
  const targetUrl = new URL(`${API_BASE_URL}${incomingUrl.pathname}${incomingUrl.search}`);
  const headers = new Headers(request.headers);

  headers.set("ngrok-skip-browser-warning", "true");
  headers.delete("host");
  headers.delete("origin");

  const response = await fetch(targetUrl, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "follow",
  });

  const responseHeaders = new Headers(response.headers);
  responseHeaders.set("Access-Control-Allow-Origin", "*");
  responseHeaders.delete("content-encoding");
  responseHeaders.delete("content-length");

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
};

export const config = {
  path: "/api/*",
};
