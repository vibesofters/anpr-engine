"use client";

import Link from "next/link";
import type { Notice } from "../../../../packages/api-contract/generated/typescript/contracts";
import { useLocale } from "./site-chrome";

type Topic = "privacy" | "responsible-use" | "limitations";

export function InfoContent({ topic, notices }: { topic: Topic; notices: { en: Notice; tr: Notice } }) {
  const locale = useLocale();
  const value = notices[locale];
  const names = locale === "tr" ? { privacy: "Gizlilik", "responsible-use": "Sorumlu kullanım", limitations: "Sınırlamalar" } : { privacy: "Privacy", "responsible-use": "Responsible use", limitations: "Limitations" };
  const paragraphs = topic === "privacy" ? [value.purpose, value.authorization, value.retention, value.training] : topic === "responsible-use" ? [value.authorization, value.prohibited_uses, value.human_review] : [value.accuracy, value.human_review, locale === "tr" ? "Bu hizmet yüksek kullanılabilirlik veya gözetimsiz üretim sistemi değildir." : "This is not a high-availability or unattended production system."];
  return <main className="shell info-page"><Link href="/" className="brand">← ANPR Engine</Link><p className="eyebrow">ANPR ENGINE / INFORMATION</p><h1>{names[topic]}</h1>{paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}<p><a href="mailto:erncnkr@gmail.com">erncnkr@gmail.com</a></p></main>;
}
