import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PROTECTED = ["/dashboard", "/exam-prep", "/career", "/learning-portal"];
const COOKIE_NAME = "mentora_session";

export function middleware(request: NextRequest) {
  const session = request.cookies.get(COOKIE_NAME)?.value;
  const { pathname } = request.nextUrl;

  const isProtected = PROTECTED.some((p) => pathname.startsWith(p));
  const isAuthPage = pathname === "/login" || pathname === "/register";

  if (isProtected && !session) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  if (isAuthPage && session) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/exam-prep/:path*",
    "/career/:path*",
    "/learning-portal/:path*",
    "/login",
    "/register",
  ],
};
