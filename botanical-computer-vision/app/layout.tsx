import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BioImages Benchmark — Three Strict Baselines",
  description: "A strict comparison of BioCLIP zero-shot, frozen BioCLIP plus a linear probe, and frozen DINOv3 plus a linear probe on the same 211 test images.",
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title: "BioImages — Three Strict Baselines",
    description: "BioCLIP probe 87.2% · BioCLIP zero-shot 85.8% · DINOv3 probe 56.4% Top-1",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "BioImages strict baseline benchmark" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "BioImages — Three Strict Baselines",
    description: "BioCLIP probe 87.2% · BioCLIP zero-shot 85.8% · DINOv3 probe 56.4% Top-1",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
