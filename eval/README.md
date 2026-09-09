# eval/

Spec Bölüm 5.3 ve 5.4'ün gerektirdiği değerlendirme katmanı.

## test_set.jsonl

**79 kayıt, elle yazıldı.** 24'ü eğitimde **GÖRÜLMEMİŞ** tool şemaları üzerine
kurulu. Eğitim verisinden ayrı tutulur, asla eğitime karıştırılmaz.

**Neden gerekli:** Modelin gerçekten "JSON şema → doğru çağrı" mantığını mı
öğrendiğini, yoksa eğitimdeki spesifik tool isimlerini mi ezberlediğini
ayırt etmenin tek yolu budur. Ölçüldü: görülmemiş şemalarda 23/24 (%95),
görülen şemalarda 44/51 (%86) — aradaki farkın **kapanmış** olması projenin
ana bulgusu.

Kategori dağılımı (`validate_test_set.py` her koşuda yazdırır):

| kategori | adet | ne ölçüyor |
|---|---|---|
| `unseen_positive` / `unseen_negative` | 11 / 5 | görülmemiş şemada doğru çağrı / gereksiz çağrı yokluğu |
| `seen_positive` / `seen_negative` | 6 / 8 | görülen şemalarda referans karşılaştırması |
| `boundary_positive` / `boundary_ambiguous` | 6 / 4 | çağrı gerektiren-gerektirmeyen sınırı |
| `wrong_tool_trap` | 5 | yakın ama yanlış tool'a sapma |
| `missing_argument` | 4 | eksik bilgide uydurma yerine sorma |
| `reasoning` | 5 | aritmetiğin `calculate`'e yönlenmesi |
| `observation_final` | 6 | gözlem sonrası final-turn dili |
| `forecast` | 5 | `days_ahead` çıkarımı |
| `explicit_search` / `query_language` | 4 / 3 | açık arama isteği / sorgu dili |
| `multi_call` / `multi_turn` | 2 / 3 | tek turda iki çağrı / takip sorusu |
| `user_language_edge` | 2 | kısa/karışık mesajda dil tespiti |

Kayıt şeması ve puanlama kuralları: **`test_set_SEMA.md`**.

Birden fazla davranışın savunulabilir olduğu kayıtlar **puanlanmaz,
raporlanır** (`scoring` alanında `report`). Metriği tamamlanmış göstermek için
referans cevap uydurmak, metriğin kendisini bozardı.

⚠️ **Üreteci kayboldu.** `uret_test_set.py` silinen bir `scratchpad/` dizinindeydi.
Artık `test_set.jsonl`'in kendisi kaynaktır — elle düzenlenebilir, ama **her
düzenlemeden sonra** `validate_test_set.py` koşulmalıdır. Doğrulayıcı bu yüzden
repoya taşındı.

## Dosyalar

- `test_set.jsonl` — 79 kayıtlık tutulmuş (held-out) set.
- `test_set_SEMA.md` — kayıt şeması + puanlama kuralları.
- `validate_test_set.py` — yapısal denetim: şema/tip/enum uyumu, zorunlu
  parametre kapsaması, rol sırası, `ipython` çift kodlaması, eğitim sızıntısı,
  eval/üretim şema kopması, `INTENTIONAL_DRIFT` denetimi. Buradaki her kontrol
  en az bir kez gerçek bir hata yakaladı.
- `eval_post_quant.py` — GGUF kuantize modeli test setine karşı puanlar.
- `eval_pre_quant.py` — bfloat16 baseline. **Sadece spec taslağı; hiç
  çalıştırılmadı ve artık çalıştırılamaz** (aşağı bkz.).

```bash
python eval/validate_test_set.py                       # önce bu
python eval/eval_post_quant.py --grammar --out r.json  # sonra bu
```

`--grammar` burada **kapalı** varsayılandır (üretimde açıktır): modelin çıplak
halini de ölçebilmek gerekiyor. Grammar her kayıt için **o kaydın kendi
tool'larından** üretilir — üretim registry'sinden değil. Sabit bir grammar,
kaydın deklare ettiği görülmemiş tool'ları yasaklar ve `unseen_positive`
0/11'e düşer; bu bir harness hatasıdır, model hatası gibi görünür.

## ❌ bf16 baseline alınamıyor

`eval_pre_quant.py` merge edilmiş bfloat16 modeli ölçmek için yazılmış bir
taslaktı. Ölçüm yapılmadan **önce** merge edilmiş model, quantize pipeline'ı
tarafından silindi. Dolayısıyla buradaki tüm sayılar **mutlak**tır,
kuantizasyon öncesine göre karşılaştırmalı değildir: "Q4_K_M ne kaybettirdi?"
sorusu bu repoda cevapsızdır.

## ⚠️ Eval ajanın tam promptunu ölçmez

`eval_post_quant.py`, `inference/prompts.py`'den **sistem promptunu** alır
(varyantlar tek kaynaktan gelsin diye), ama orchestrator'ın üretimden hemen
önce bastığı **son-an direktifini eklemez**. Yani buradaki `final_language` ve
`observation_final` sayıları, dili asıl zorlayan mekanizma **yokken** ölçülmüş
ham model davranışıdır. Bilinçli bir sınır: eval modelin kendi eğilimini
ölçüyor, ajanın davranışını değil. Direktifi eklemek tüm ölçümlerin yeniden
koşulmasını gerektirir.
