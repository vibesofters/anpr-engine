"use client";
/* eslint-disable @next/next/no-img-element -- The current-session blob URL cannot use server image optimization. */

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import type { ChangeEvent, FormEvent } from "react";
import { useLocale } from "./site-chrome";
import type {
  AnprApiErrorV1, AnprImageInferenceResponseV1, AnprSessionResultV1, Notice
} from "../../../../packages/api-contract/generated/typescript/contracts";

type Review = "OWNER_CONFIRMED_PROJECT_MODEL" | "OWNER_CORRECTED" | "UNRESOLVED" | null;
const LIMIT = 10 * 1024 * 1024;
const CLIENT_DEADLINE_MS = 140_000;
const labels = {
  en: {
    heroLabel: "ANPR ENGINE / PUBLIC DEMONSTRATION", limits: "JPEG / PNG · ≤ 10 MiB · ≤ 20 MP · ≤ 8,192 px per side", home: "ANPR Engine home", information: "Information",
    tagline: "A single-image ANPR demonstration", intro: "Project-owned Detection and Recognition models. Results can be wrong and always need your review.",
    stepNotice: "01 / Review the notice", stepUpload: "02 / Choose one image", stepResult: "03 / Review the result",
    select: "Choose a JPEG or PNG", selected: "Selected image", submit: "Process image", processing: "Processing one image…", clear: "Clear current session",
    original: "Original image with Detection box", detection: "Detection", recognition: "Recognition prediction", structure: "Structural validation", confidence: "Informational confidence, not certainty",
    noPlate: "No plate was detected. Try a different authorized image.", unresolved: "This result is unresolved until you review it.",
    confirm: "Confirm prediction", correct: "Correct plate text", correction: "Corrected plate text", applyCorrection: "Use correction", leaveUnresolved: "Leave unresolved",
    confirmed: "You confirmed the model prediction.", corrected: "You corrected the text.", reviewUnresolved: "The result remains unresolved.",
    json: "Download versioned JSON", annotated: "Download annotated image", unavailable: "An annotated image needs a Detection box.",
    found: "Plate area detected", notFound: "No plate area detected", valid: "Format check passed", invalid: "Format check failed", notRun: "Format check not run",
    privacy: "Privacy", responsible: "Responsible use", limitations: "Limitations", language: "Language", contact: "Contact", repo: "Repository",
    invalidFile: "Choose one complete, single-frame JPEG or PNG within 10 MiB, 20 megapixels and 8,192 pixels per side.",
    acknowledge: "Acknowledge the notice before processing.", busy: "Only one image can be processed at a time.", error: "The image could not be processed. Review it or try again manually.",
    rate: "Limit reached: at most 5 valid processing attempts per IP each day (Istanbul time). Shared networks share this limit.", capacity: "The demonstration is busy or temporarily unavailable. Try again later.", timeout: "Processing timed out. Please try again later.", unavailableService: "Image processing is temporarily unavailable.",
    precision: "The box marks the Detection result on your original image; it is not the model's refined crop.",
    warning: "Neither confidence nor format validation proves that the plate text is correct.",
    multipleWarning: "Multiple plate areas were found. Only the highest-ranked Detection was processed.", cropWarning: "The detected area could not be refined or read. This result is unresolved.", recognitionWarning: "Recognition did not produce a usable prediction. This result is unresolved.",
    downloadNote: "Downloads are created only when you click and are not saved by this site."
  },
  tr: {
    heroLabel: "ANPR ENGINE / KAMUYA AÇIK GÖSTERİM", limits: "JPEG / PNG · ≤ 10 MiB · ≤ 20 MP · kenar başına ≤ 8.192 piksel", home: "ANPR Engine ana sayfa", information: "Bilgilendirme",
    tagline: "Tek görüntülü ANPR gösterimi", intro: "Projeye ait Tespit ve Tanıma modelleri. Sonuçlar yanlış olabilir ve her zaman incelemeniz gerekir.",
    stepNotice: "01 / Bildirimi inceleyin", stepUpload: "02 / Bir görüntü seçin", stepResult: "03 / Sonucu inceleyin",
    select: "JPEG veya PNG seçin", selected: "Seçilen görüntü", submit: "Görüntüyü işle", processing: "Bir görüntü işleniyor…", clear: "Geçerli oturumu temizle",
    original: "Tespit kutulu özgün görüntü", detection: "Tespit", recognition: "Tanıma tahmini", structure: "Yapısal doğrulama", confidence: "Bilgilendirici güven puanı; kesinlik değil",
    noPlate: "Plaka tespit edilmedi. Yetkili başka bir görüntü deneyin.", unresolved: "Siz inceleyene kadar bu sonuç çözümlenmemiştir.",
    confirm: "Tahmini onayla", correct: "Plaka metnini düzelt", correction: "Düzeltilmiş plaka metni", applyCorrection: "Düzeltmeyi uygula", leaveUnresolved: "Çözümlenmemiş bırak",
    confirmed: "Model tahminini onayladınız.", corrected: "Metni düzelttiniz.", reviewUnresolved: "Sonuç çözümlenmemiş olarak kaldı.",
    json: "Sürümlü JSON indir", annotated: "İşaretli görüntüyü indir", unavailable: "İşaretli görüntü için Tespit kutusu gereklidir.",
    found: "Plaka alanı tespit edildi", notFound: "Plaka alanı tespit edilmedi", valid: "Biçim denetimi geçti", invalid: "Biçim denetimi başarısız", notRun: "Biçim denetimi yapılmadı",
    privacy: "Gizlilik", responsible: "Sorumlu kullanım", limitations: "Sınırlamalar", language: "Dil", contact: "İletişim", repo: "Depo",
    invalidFile: "10 MiB, 20 megapiksel ve kenar başına 8.192 piksel sınırları içinde tek, tam, tek kareli JPEG veya PNG seçin.",
    acknowledge: "İşlemeden önce bildirimi onaylayın.", busy: "Aynı anda yalnızca bir görüntü işlenebilir.", error: "Görüntü işlenemedi. İnceleyin veya elle yeniden deneyin.",
    rate: "Günlük sınır doldu: IP adresi başına en fazla 5 geçerli işleme denemesi (İstanbul saati). Ortak ağlarda bu sınır paylaşılır.", capacity: "Gösterim meşgul veya geçici olarak kullanılamıyor. Daha sonra deneyin.", timeout: "İşlem zaman aşımına uğradı. Daha sonra tekrar deneyin.", unavailableService: "Görüntü işleme geçici olarak kullanılamıyor.",
    precision: "Kutu, özgün görüntüdeki Tespit sonucunu gösterir; modelin iyileştirilmiş kırpması değildir.",
    warning: "Güven puanı veya biçim denetimi plaka metninin doğru olduğunu kanıtlamaz.",
    multipleWarning: "Birden fazla plaka alanı bulundu. Yalnızca en üst sıradaki Tespit işlendi.", cropWarning: "Tespit edilen alan iyileştirilemedi veya okunamadı. Sonuç çözümlenmemiştir.", recognitionWarning: "Tanıma kullanılabilir bir tahmin üretmedi. Sonuç çözümlenmemiştir.",
    downloadNote: "İndirmeler yalnızca tıklamanızla oluşturulur ve bu site tarafından saklanmaz."
  }
} as const;

function downloadable(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function errorMessage(code: string | undefined, t: typeof labels.en | typeof labels.tr): string {
  if (code === "RATE_LIMITED") return t.rate;
  if (code === "CAPACITY_UNAVAILABLE" || code === "SESSION_INFERENCE_ACTIVE") return t.capacity;
  if (code === "INFERENCE_DEADLINE_EXCEEDED") return t.timeout;
  if (code === "MODEL_NOT_READY" || code === "MAINTENANCE") return t.unavailableService;
  if (code === "ORIGIN_REJECTED") return t.unavailableService;
  if (code === "PRIVACY_ACKNOWLEDGEMENT_REQUIRED") return t.acknowledge;
  if (code?.startsWith("IMAGE_") || code === "UNSUPPORTED_MEDIA_TYPE") return t.invalidFile;
  return t.error;
}

export function Workspace({ notices, inferenceAvailable }: { notices: { en: Notice; tr: Notice }; inferenceAvailable: boolean }) {
  const locale = useLocale();
  const notice = notices[locale];
  const t = labels[locale];
  const [acknowledged, setAcknowledged] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnprImageInferenceResponseV1 | null>(null);
  const [review, setReview] = useState<Review>(null);
  const [corrected, setCorrected] = useState("");
  const [reviewedValue, setReviewedValue] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const busyRef = useRef(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => { controller.current?.abort(); if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function clear(preserveAcknowledgement = false) {
    if (busyRef.current) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setResult(null); setReview(null); setCorrected(""); setReviewedValue(null); setMessage("");
    if (!preserveAcknowledgement) setAcknowledged(false);
  }

  async function select(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0];
    event.target.value = "";
    if (!chosen || busyRef.current) return;
    clear(true);
    const extension = chosen.name.split(".").pop()?.toLowerCase();
    if (!((chosen.type === "image/jpeg" && ["jpg", "jpeg"].includes(extension ?? "")) ||
          (chosen.type === "image/png" && extension === "png")) || chosen.size === 0 || chosen.size > LIMIT) {
      setMessage(t.invalidFile); return;
    }
    try {
      const bitmap = await createImageBitmap(chosen);
      const valid = bitmap.width <= 8192 && bitmap.height <= 8192 && bitmap.width * bitmap.height <= 20_000_000;
      bitmap.close();
      if (!valid) { setMessage(t.invalidFile); return; }
      setFile(chosen); setPreview(URL.createObjectURL(chosen));
    } catch { setMessage(t.invalidFile); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busyRef.current) { setMessage(t.busy); return; }
    if (!acknowledged) { setMessage(t.acknowledge); return; }
    if (!file) { setMessage(t.invalidFile); return; }
    if (!inferenceAvailable) { setMessage(t.unavailableService); return; }
    busyRef.current = true; setBusy(true); setMessage(""); setResult(null); setReview(null); setReviewedValue(null);
    const abort = new AbortController(); controller.current = abort;
    const deadline = window.setTimeout(() => abort.abort(), CLIENT_DEADLINE_MS);
    try {
      const response = await fetch("/api/inference", {
        method: "POST", credentials: "same-origin", cache: "no-store", signal: abort.signal,
        headers: { "Content-Type": file.type, "X-Privacy-Acknowledged": "true" }, body: file
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        const code = (body as AnprApiErrorV1 | null)?.error?.code;
        setMessage(errorMessage(code, t)); return;
      }
      const value = body as AnprImageInferenceResponseV1;
      if (value.schema_version !== "anpr-image-inference-response-v1" || value.human_review_required !== true) {
        setMessage(t.error); return;
      }
      setResult(value);
      if (value.status === "no_plate_detected") setReview("UNRESOLVED");
    } catch { setMessage(abort.signal.aborted ? t.timeout : t.error); }
    finally { window.clearTimeout(deadline); busyRef.current = false; setBusy(false); controller.current = null; }
  }

  function downloadJson() {
    if (!result) return;
    const payload: AnprSessionResultV1 = {
      schema_version: "anpr-session-result-v1", result_schema_version: "anpr-image-inference-response-v1",
      request_id: result.request_id, review_outcome: review ?? "UNRESOLVED",
      model_prediction: result.recognition.prediction,
      reviewed_value: review === "OWNER_CORRECTED" ? reviewedValue : review === "OWNER_CONFIRMED_PROJECT_MODEL" ? result.recognition.prediction : null,
      warnings: result.warnings
    };
    downloadable(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), "anpr-result-v1.json");
  }

  function downloadAnnotated() {
    const box = result?.detection.box_xyxy_normalized;
    const image = imageRef.current;
    if (!box || box.length !== 4 || !image?.complete || !image.naturalWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#e85d3f"; context.lineWidth = Math.max(3, canvas.width / 300);
    context.strokeRect(box[0] * canvas.width, box[1] * canvas.height, (box[2] - box[0]) * canvas.width, (box[3] - box[1]) * canvas.height);
    canvas.toBlob((blob) => { if (blob) downloadable(blob, "anpr-detection-overlay-v1.png"); }, "image/png");
  }

  const box = result?.detection.box_xyxy_normalized;
  return <main className="shell">
      <section className="hero"><div className="hero-copy"><p className="eyebrow">{t.heroLabel}</p><h1>{t.tagline}</h1><p className="lead">{t.intro}</p><div className="hero-detail"><span className="status-dot" />{locale === "en" ? "One image at a time · Human review required" : "Bir seferde bir görüntü · İnsan incelemesi gerekli"}</div><ol className="hero-flow" aria-label={locale === "en" ? "How it works" : "Nasıl çalışır"}><li><span>01</span>{locale === "en" ? "Detect" : "Tespit"}</li><li><span>02</span>{locale === "en" ? "Recognize" : "Tanıma"}</li><li><span>03</span>{locale === "en" ? "You review" : "Siz inceleyin"}</li></ol></div><div className="hero-art" aria-hidden="true"><div className="hero-art-halo" /><Image src="/icon.png" alt="" width={250} height={250} priority /><span>PROJECT-OWNED MODELS</span></div></section>
      <div className="workflow">
        <section className="panel" aria-labelledby="notice-title"><p className="step">{t.stepNotice}</p><h2 id="notice-title">{locale === "en" ? "Processing & responsible use" : "İşleme ve sorumlu kullanım"}</h2><div className="notice-text"><p>{notice.purpose}</p><p>{notice.authorization}</p><p>{notice.accuracy} {notice.human_review}</p><p>{notice.retention} {notice.training}</p><p>{notice.prohibited_uses}</p></div><label className="checkbox-row"><input type="checkbox" checked={acknowledged} disabled={busy} onChange={(event) => setAcknowledged(event.target.checked)} /><span>{notice.acknowledgement_label}</span></label></section>
      <section className="panel" aria-labelledby="upload-title"><p className="step">{t.stepUpload}</p><h2 id="upload-title">{t.select}</h2><p className="subtle">{t.limits}</p><form onSubmit={submit}><label className={`file-picker ${busy ? "disabled" : ""}`}><span className="file-icon" aria-hidden="true">＋</span><span>{file ? `${t.selected}: ${file.name}` : t.select}</span><input type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" disabled={busy || !inferenceAvailable} onChange={select} aria-label={t.select} /></label><div className="action-row"><button type="submit" className="primary" disabled={busy || !file || !acknowledged || !inferenceAvailable}>{busy ? t.processing : t.submit}</button><button type="button" className="secondary" onClick={() => clear()} disabled={busy || (!file && !result)}>{t.clear}</button></div></form><p role="status" aria-live="polite" className="feedback">{!inferenceAvailable ? t.unavailableService : message || (busy ? t.processing : "")}</p></section>
        <section className="panel result-panel" aria-labelledby="result-title"><p className="step">{t.stepResult}</p><h2 id="result-title">{locale === "en" ? "Result & human review" : "Sonuç ve insan incelemesi"}</h2>{!result ? <div className="empty-state">{locale === "en" ? "Your current result will appear here after processing." : "Geçerli sonuç işlemden sonra burada görünecek."}</div> : <>
          {preview && <figure><div className="image-frame"><img ref={imageRef} src={preview} alt={t.original} />{box?.length === 4 && <span className="detection-box" aria-label={t.found} style={{ left: `${box[0]*100}%`, top: `${box[1]*100}%`, width: `${(box[2]-box[0])*100}%`, height: `${(box[3]-box[1])*100}%` }} />}</div><figcaption>{t.precision}</figcaption></figure>}
          <div className="readout"><div><span className="readout-label">{t.detection}</span><strong>{result.detection.state === "found" ? t.found : t.notFound}</strong></div><div><span className="readout-label">{t.recognition}</span><strong className="plate-text">{result.recognition.prediction ?? "—"}</strong></div><div><span className="readout-label">{t.structure}</span><strong>{result.structural_validation === "valid" ? t.valid : result.structural_validation === "invalid" ? t.invalid : t.notRun}</strong></div></div>
          {result.status === "no_plate_detected" && <p className="feedback">{t.noPlate}</p>}
          {result.warnings.includes("MULTIPLE_DETECTIONS_TOP1_SELECTED") && <p className="feedback">{t.multipleWarning}</p>}
          {result.warnings.includes("CROP_REFINEMENT_FAILED") && <p className="feedback">{t.cropWarning}</p>}
          {result.warnings.includes("RECOGNITION_UNRESOLVED") && <p className="feedback">{t.recognitionWarning}</p>}
          <p className="caution">{t.warning}</p><p className="subtle">{t.confidence}: {result.recognition.confidence === null ? "—" : `${Math.round(result.recognition.confidence * 100)}%`}</p>
          <div className="review"><h3>{locale === "en" ? "Your review" : "İncelemeniz"}</h3><p>{review === "OWNER_CONFIRMED_PROJECT_MODEL" ? t.confirmed : review === "OWNER_CORRECTED" ? t.corrected : review === "UNRESOLVED" ? t.reviewUnresolved : t.unresolved}</p><div className="action-row"><button type="button" className="secondary" onClick={() => setReview("OWNER_CONFIRMED_PROJECT_MODEL")} disabled={!result.recognition.prediction}>{t.confirm}</button><button type="button" className="secondary" onClick={() => setReview("UNRESOLVED")}>{t.leaveUnresolved}</button></div><label className="correction-label">{t.correction}<input type="text" inputMode="text" maxLength={16} value={corrected} onChange={(event) => { setCorrected(event.target.value.toUpperCase().replace(/[^0-9A-Z ]/g, "")); if (review === "OWNER_CORRECTED") setReview(null); }} aria-label={t.correction} /></label><button type="button" className="secondary" disabled={!/^[0-9A-Z](?:[0-9A-Z ]{0,14}[0-9A-Z])?$/.test(corrected)} onClick={() => { setReviewedValue(corrected); setReview("OWNER_CORRECTED"); }}>{t.applyCorrection}</button></div>
          <div className="downloads"><button type="button" className="primary" onClick={downloadJson}>{t.json}</button><button type="button" className="secondary" onClick={downloadAnnotated} disabled={!box}>{t.annotated}</button>{!box && <p className="subtle">{t.unavailable}</p>}<p className="subtle">{t.downloadNote}</p></div>
        </>}</section>
      </div>
    </main>;
}
