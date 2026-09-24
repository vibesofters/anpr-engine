import type { ReactNode } from "react";
import { cookies } from "next/headers";
import { SiteChrome } from "../components/site-chrome";
import "./styles.css";

export default async function RootLayout({ children }: { children: ReactNode }) {
  const theme = (await cookies()).get("anpr_theme")?.value === "light" ? "light" : "dark";
  return <html lang="en" data-theme={theme}><body><SiteChrome initialTheme={theme}>{children}</SiteChrome></body></html>;
}
