import { NextRequest, NextResponse } from "next/server";
import { fastApiFetch, ApiError } from "@/lib/api";

export async function POST(request: NextRequest) {
  const body = await request.json();
  try {
    const data = await fastApiFetch("/auth/reset-password", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (err) {
    const status = err instanceof ApiError ? err.status : 500;
    const message = err instanceof ApiError ? err.message : "Reset failed";
    return NextResponse.json({ detail: message }, { status });
  }
}
