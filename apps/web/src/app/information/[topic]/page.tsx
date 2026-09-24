import type { Metadata } from "next";
import { notFound } from "next/navigation";
import notice from "../../../../../../packages/api-contract/fixtures/valid/processing-notice.json";
import { InfoContent } from "@/components/info-content";

type Props = { params: Promise<{ topic: string }> };
const topics = ["privacy", "responsible-use", "limitations"] as const;

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { topic } = await params;
  if (!topics.includes(topic as typeof topics[number])) return {};
  return { title: `${topic.replaceAll("-", " ")} · ANPR Engine`, alternates: { canonical: `https://anpr-engine.vibesofters.com/information/${topic}` }, robots: { index: false, follow: false } };
}

export default async function Topic({ params }: Props) {
  const { topic } = await params;
  if (!topics.includes(topic as typeof topics[number])) notFound();
  return <InfoContent topic={topic as typeof topics[number]} notices={notice.notices} />;
}
