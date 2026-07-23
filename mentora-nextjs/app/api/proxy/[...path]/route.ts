import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { COOKIE_NAME } from "@/lib/api";

// Client Components call /api/proxy/<fastapi-path> instead of FastAPI
// directly. This route reads the httpOnly JWT server-side, attaches it
// as `Authorization: Bearer <token>`, and forwards the request —
// per MENTORA_Phase1_Frontend_Documentation.md §10.3 step 6: "route
// through a small Route Handler rather than calling FastAPI directly
// from the browser."
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function forward(request: NextRequest, path: string[]) {
  const token = cookies().get(COOKIE_NAME)?.value;
  const targetPath = "/" + path.join("/");
  const search = request.nextUrl.search;

  const isFormData = (request.headers.get("content-type") || "").includes("multipart/form-data");

  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!isFormData) headers["Content-Type"] = "application/json";

  const init: RequestInit = {
    method: request.method,
    headers,
    cache: "no-store",
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = isFormData ? await request.formData() : await request.text();
  }

  const res = await fetch(`${API_URL}${targetPath}${search}`, init as RequestInit & { duplex?: "half" });
  const contentType = res.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await res.json() : await res.text();

  return NextResponse.json(data, { status: res.status });
}

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(request, params.path);
}
export async function POST(request: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(request, params.path);
}
export async function PATCH(request: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(request, params.path);
}
export async function DELETE(request: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(request, params.path);
}
