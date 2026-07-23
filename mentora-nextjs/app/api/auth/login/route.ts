import { NextRequest, NextResponse } from "next/server";
import { fastApiFetch, COOKIE_NAME, ApiError } from "@/lib/api";
import type { AuthResponse } from "@/lib/types";

// This is the ONLY place the raw JWT ever touches Next.js's server before
// being sealed into an httpOnly cookie — see the frontend doc §10.3.
export async function POST(request: NextRequest) {
  const body = await request.json();

  try {
    const data = await fastApiFetch<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    });

    const response = NextResponse.json({ success: true, user: data.user });
    response.cookies.set(COOKIE_NAME, data.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production", // never hardcode true — breaks local http
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 7, // 7 days, matches backend JWT_EXPIRE_MINUTES default
    });
    return response;
  } catch (err) {
    const status = err instanceof ApiError ? err.status : 500;
    const message = err instanceof ApiError ? err.message : "Login failed";
    return NextResponse.json({ detail: message }, { status });
  }
}
