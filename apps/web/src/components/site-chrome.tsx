"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { flushSync } from "react-dom";

export type Locale = "en" | "tr";
type Theme = "dark" | "light";
const LocaleContext = createContext<Locale>("en");

export function useLocale(): Locale { return useContext(LocaleContext); }

const copy = {
  en: { home: "ANPR Engine home", navigation: "Contact, language and appearance", language: "Switch to Turkish", contactUs: "Contact us", light: "Light", dark: "Dark", switchLight: "Switch to light theme", switchDark: "Switch to dark theme", information: "Information", privacy: "Privacy", responsible: "Responsible use", limitations: "Limitations", repo: "Repository", contact: "Contact" },
  tr: { home: "ANPR Engine ana sayfa", navigation: "İletişim, dil ve görünüm", language: "İngilizceye geç", contactUs: "İletişime Geç", light: "Açık", dark: "Koyu", switchLight: "Açık temaya geç", switchDark: "Koyu temaya geç", information: "Bilgilendirme", privacy: "Gizlilik", responsible: "Sorumlu kullanım", limitations: "Sınırlamalar", repo: "Depo", contact: "İletişim" }
} as const;

const pageTitles = {
  en: { home: "ANPR Engine — Plate detection and recognition demonstration", privacy: "Privacy · ANPR Engine", "responsible-use": "Responsible use · ANPR Engine", limitations: "Limitations · ANPR Engine" },
  tr: { home: "ANPR Engine — Plaka tespit ve tanıma gösterimi", privacy: "Gizlilik · ANPR Engine", "responsible-use": "Sorumlu kullanım · ANPR Engine", limitations: "Sınırlamalar · ANPR Engine" }
} as const;

export function SiteChrome({ initialTheme, children }: { initialTheme: Theme; children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const pathname = usePathname();
  const t = copy[locale];

  useEffect(() => {
    const topic = pathname.split("/").at(-1);
    document.documentElement.lang = locale;
    document.title = topic === "privacy" || topic === "responsible-use" || topic === "limitations" ? pageTitles[locale][topic] : pageTitles[locale].home;
  }, [locale, pathname]);

  function toggleLanguage() {
    const next = locale === "en" ? "tr" : "en";
    const applyLanguage = () => {
      document.documentElement.lang = next;
      flushSync(() => setLocale(next));
    };
    if (document.startViewTransition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      document.startViewTransition(applyLanguage);
    } else {
      applyLanguage();
    }
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    const applyTheme = () => {
      document.documentElement.dataset.theme = next;
      document.cookie = `anpr_theme=${next}; Path=/; Max-Age=31536000; SameSite=Strict${location.protocol === "https:" ? "; Secure" : ""}`;
      flushSync(() => setTheme(next));
    };

    if (document.startViewTransition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      document.startViewTransition(applyTheme);
    } else {
      applyTheme();
    }
  }

  return <LocaleContext.Provider value={locale}>
    <header className="site-header"><div className="shell header-inner">
      <Link className="brand" href="/" aria-label={t.home}><Image src="/icon.png" alt="" width={38} height={38} priority /><span>ANPR <span className="brand-accent">Engine</span></span></Link>
      <nav className="header-controls" aria-label={t.navigation}>
        <a className="header-control contact-control" href="https://vibesofters.com">{t.contactUs}</a>
        <button className="header-control language-control" type="button" onClick={toggleLanguage} lang={locale === "en" ? "tr" : "en"} aria-label={t.language}>{locale === "en" ? "Türkçe" : "English"}</button>
        <button className="header-control theme-control" type="button" onClick={toggleTheme} aria-label={theme === "dark" ? t.switchLight : t.switchDark} title={theme === "dark" ? t.switchLight : t.switchDark}>
          <span className="theme-symbol" aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span><span>{theme === "dark" ? t.light : t.dark}</span>
        </button>
      </nav>
    </div></header>
    {children}
    <footer className="site-footer"><div className="shell footer-inner"><span>© 2026 VibeSofters - Eren Cankur · M. Sırac Özer · M. Akif Erdurur</span><nav aria-label={t.information}><Link href="/information/privacy">{t.privacy}</Link><Link href="/information/responsible-use">{t.responsible}</Link><Link href="/information/limitations">{t.limitations}</Link><a href="https://github.com/vibesofters/anpr-engine" rel="noreferrer">{t.repo}</a><a href="mailto:erncnkr@gmail.com">{t.contact}</a></nav></div></footer>
  </LocaleContext.Provider>;
}
