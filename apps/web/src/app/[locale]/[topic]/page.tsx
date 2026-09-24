import { notFound, redirect } from "next/navigation";

type Props = { params: Promise<{ locale: string; topic: string }> };
const topics = ["privacy", "responsible-use", "limitations"] as const;

export default async function Topic({ params }: Props) {
  const { locale, topic } = await params;
  if ((locale !== "en" && locale !== "tr") || !topics.includes(topic as typeof topics[number])) notFound();
  redirect(`/information/${topic}`);
}
