import { NextResponse } from "next/server";
import { COOKIE_NAME } from "@/lib/api";

export async function POST() {
  const response = NextResponse.json({ success: true });
  response.cookies.set(COOKIE_NAME, "", { path: "/", maxAge: 0 });
  return response;
}
