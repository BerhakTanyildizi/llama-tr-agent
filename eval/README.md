# eval/

Spec Bölüm 5.3 ve 5.4'ün gerektirdiği değerlendirme katmanı.

## test_set.jsonl

**Eğitimde GÖRÜLMEMİŞ tool şemaları** içeren genelleme test seti. Eğitim
verisinden ayrı tutulur, asla eğitime karıştırılmaz.

İçermesi gerekenler:

- Mevcut üç tool'a (`google_search`, `get_weather`, `get_system_time`) yapıda
  benzeyen ama **farklı isimli ve farklı parametreli** yeni tool'lar
  (örn. dördüncü, beşinci bir tool).
- Referans karşılaştırması için eğitimde görülen tool'lardan da bir alt küme.
- Tool çağrısı GEREKTİRMEYEN, doğrudan cevaplanması gereken örnekler
  (gereksiz tool çağrısı hatasını yakalamak için).
- Türkçe final-turn beklenen örnekler (dil doğrulaması için).

**Neden gerekli:** Modelin gerçekten "JSON şema → doğru çağrı" mantığını mı
öğrendiğini, yoksa 10K örnekteki spesifik tool isimlerini mi ezberlediğini
ayırt etmenin tek yolu budur.

## Dosyalar

- `test_set.jsonl` — test seti (şu an boş).
- `eval_pre_quant.py` — merge edilmiş bfloat16 model ölçümü (baseline).
- `eval_post_quant.py` — GGUF kuantize model ölçümü + baseline karşılaştırması.
