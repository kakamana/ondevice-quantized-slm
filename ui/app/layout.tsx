import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "On-Device SLM Summarization",
  description: "Pareto-aware summarization for edge devices",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
