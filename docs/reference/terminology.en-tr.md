# English/Turkish terminology foundation

English is authoritative during development. Turkish wording is provisional and
requires native-language review before public launch. Model IDs, SHA-256 values,
schema names, API fields, CLI options, and release identifiers remain unchanged.

| Key | English | Turkish — native review required |
|---|---|---|
| `upload_image` | Upload image | Görüntü yükle |
| `authorized_image` | Authorized image | Yetkili görüntü |
| `processing_notice` | Processing notice | İşleme bildirimi |
| `detection` | Detection | Tespit |
| `recognition` | Recognition | Tanıma |
| `crop_refinement` | Crop refinement | Kırpma iyileştirme |
| `prediction` | Prediction | Tahmin |
| `structural_validation` | Structural validation | Yapısal doğrulama |
| `human_review` | Human review | İnsan incelemesi |
| `confirm` | Confirm | Onayla |
| `correct` | Correct | Düzelt |
| `resolved` | Resolved | Çözümlenmiş |
| `unresolved` | Unresolved | Çözümlenmemiş |
| `processing` | Processing | İşleniyor |
| `no_plate_detected` | No plate detected | Plaka tespit edilmedi |
| `image_rejected` | Image rejected | Görüntü reddedildi |
| `busy` | Busy | Meşgul |
| `rate_limited` | Rate limited | İstek sınırına ulaşıldı |
| `model_unavailable` | Model unavailable | Model kullanılamıyor |
| `timeout` | Timeout | Zaman aşımı |
| `maintenance` | Maintenance | Bakım |
| `privacy_notice` | Privacy notice | Gizlilik bildirimi |
| `responsible_use` | Responsible use | Sorumlu kullanım |
| `clear_result` | Clear result | Sonucu temizle |
| `download_result` | Download result | Sonucu indir |

Structural validation is only a format check and does not establish identity.
Predictions remain unresolved until explicit human review. The machine-readable
seed is
[`packages/api-contract/glossary.seed.json`](../../packages/api-contract/glossary.seed.json).
