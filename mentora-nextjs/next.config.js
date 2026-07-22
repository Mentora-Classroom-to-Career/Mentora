/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The FastAPI backend runs on http://localhost:8000 during local dev
  // (see MENTORA technical docs, Part 11). Requests from the frontend
  // should be pointed at process.env.NEXT_PUBLIC_API_URL.
};

module.exports = nextConfig;
