import type { Metadata } from "next";

export const metadata: Metadata = { title: "Exam Prep" };

export default function ExamPrepLayout({ children }: { children: React.ReactNode }) {
  return children;
}
