# tests/

Harness testleri. **Sunucu, ağ ve harici bağımlılık gerektirmez** — saniyeler sürer.

```bash
python -m unittest discover -s tests -v      # hepsi
python -m unittest tests.test_grammar -v     # tek modül
python -m unittest tests.test_tools.Calculate -v
```

## Neden var

`eval/` **modeli** ölçüyor. **Harness'ı** ölçen hiçbir şey yoktu — ve bu projede
bulunan hataların çoğu harness hatası:

| hata | nasıl göründü |
|---|---|
| Parser ilk çağrıyı alıp kalanını sessizce attı | *"2 arama yap"* tek arama yaptı → model hatası sanıldı |
| Grammar kökü yalnızca `<tool_call>` kabul ediyordu | Düz cevap üretilemiyordu → model "gereksiz tool çağırıyor" sanıldı |
| Eval grammar'ı sabit dosyadan okuyordu | `unseen_positive` 0/11 → genelleme başarısız sanıldı |
| `prose ::= [^<]+` | Model `3 < 5` yerine `3 ≤ 5` yazdı → model yanlış sanıldı |
| Opsiyonel argümandaki yer tutucu çağrıyı engelledi | Doğru `calculate` çağrısı reddedildi |
| Eval `arguments` string olunca karakterleri dolaştı | `extra:1, extra:7, extra:*` |

**Hepsi model hatası gibi göründü, hiçbiri değildi.** Buradaki her test, en az bir
kez gerçekten yaşanmış bir hatayı kilitliyor.

## Modüller

| dosya | kapsam |
|---|---|
| `test_orchestrator.py` | çağrı ayrıştırma (tek/çoklu/bozuk), dispatch, grounding, zorunlu-opsiyonel yer tutucu ayrımı, bağlam budama, direktifin konumu, uyarı desenleri |
| `test_prompts.py` | dil çözümleme önceliği (`en`/`tr`/`auto`/`None`), direktif maddeleri, sistem promptunun dil cümlesi, **eval varsayılanının değişmediği**, locale-bağımsız tarih |
| `test_tools.py` | registry sözleşmesi, şema doğrulama (enum, bool tuzağı), `dispatch` asla exception atmaz, `calculate` güvenliği (`eval()` yok), locale-bağımsız gün adı, arama yardımcıları |
| `test_grammar.py` | kökün düz cevaba ve çoklu çağrıya izin vermesi, `prose`'un `<` kabul etmesi, çağrı yerinin tool demetinden üretim, tanımsız kural yokluğu, `tool_call.gbnf` sapması |

## Kurallar

- **Ağ yok.** `get_weather` yalnızca HTTP'den **önce** dönen dallarda test edilir
  (`days_ahead=99`, geçersiz tip). `google_search` yalnızca saf yardımcılarıyla.
  Ağ isteyen bir test CI'da atlanır ve hiçbir şeyi korumaz.
- **Model yok.** `FakeModel` senaryolanmış üretimleri tekrar oynatır; döngü
  gerçekten koşar, model koşmaz.
- Her testin adı neyi koruduğunu söyler; gerekçe docstring'de.

## `tool_call.gbnf` başarısız olursa

Anlık görüntü registry'den geri kalmıştır:

```bash
python inference/grammar/generate_gbnf.py
```
