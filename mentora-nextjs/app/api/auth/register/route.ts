import { NextRequest, NextResponse } from "next/server";
import { fastApiFetch, COOKIE_NAME, ApiError } from "@/lib/api";
import type { AuthResponse } from "@/lib/types";

export async function POST(request: NextRequest) {
  const body = await request.json();

  try {
    const data = await fastApiFetch<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(body),
    });

    const response = NextResponse.json({ success: true, user: data.user });
    response.cookies.set(COOKIE_NAME, data.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 7,
    });
    return response;
  } catch (err) {
    const status = err instanceof ApiError ? err.status : 500;
    const message = err instanceof ApiError ? err.message : "Registration failed";
    return NextResponse.json({ detail: message }, { status });
  }
}
