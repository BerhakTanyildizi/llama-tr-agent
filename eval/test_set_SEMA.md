# `test_set.jsonl` — şema ve puanlama kuralları

`eval_pre_quant.py` ve `eval_post_quant.py` bu dosyadaki kurallara göre puanlar.
İkisi de **aynı** kuralları uygulamak zorunda; aksi halde kuantizasyon öncesi/sonrası
karşılaştırma geçersiz olur.

⚠️ **Üreteç KAYBOLDU.** `scratchpad/uret_test_set.py` ve ilk doğrulayıcı, oturumla
birlikte silinen bir dizindeydi. Artık **`test_set.jsonl`'in kendisi kaynaktır**:
elle düzenlenir ve her düzenlemeden sonra `python eval/validate_test_set.py`
koşulur. Doğrulayıcı bu yüzden repoya taşındı.

## Kayıt şeması

```jsonc
{
  "id": "F03",
  "category": "unseen_positive",
  "tools_seen": false,          // tools listesi yalnızca eğitimdeki üç tool'dan mı ibaret
  "tools": [ /* tam JSON şemaları */ ],
  "messages": [ {"role": "user|assistant|ipython", "content": "..."} ],
  "expect": { /* aşağıya bak */ },
  "scoring": { "<alan>": "strict" | "report" },
  "note": "bu kaydın neyi ölçtüğü ve referans değerin gerekçesi"
}
```

**Sistem promptu kayıtlarda YOK.** Aynı set farklı prompt varyantlarıyla (V0/V1/V2)
koşulabilsin diye. Eval scripti promptu kendisi kurar ve `tools` alanını
`<tools></tools>` arasına basar. Eğitimdeki gövde birebir korunmalı; `Today Date`
satırına **gerçek tarih** geçilmeli (CLAUDE.md md. 6).

## `expect` alanları

| alan | anlamı |
|---|---|
| `tool_call` | Beklenen tool adı; `null` = hiç çağrı olmamalı; `"either"` = iki davranış da kabul; `"prefer_none"` = ideali çağırmamak ama çağırması hata sayılmaz; `"any_of"` = `any_of` listesinden biri |
| `args.must` | `{argüman: [kabul edilen değerler]}`. `"*"` = boş olmayan herhangi bir değer |
| `args.may` | Varlığı hata değil, yokluğu da hata değil |
| `args.if_present` | Vermek zorunlu değil, ama **verilirse** listedeki değerlerden biri olmalı |
| `final_language` | Beklenen final cevap dili (`"tr"`) |
| `must_not_fabricate_args` | Argüman değerleri konuşmadan türetilebilir olmalı |
| `must_not_fabricate_content` | Gözlemde olmayan bilgi cevaba eklenmemeli |
| `args_lang` | `{argüman: dil}` — argüman değerinin beklenen dili |
| `expected_call_count` | Beklenen çağrı sayısı |
| `args_contains` | `{argüman: [alt dize]}` — üretilen değer bunlardan **birini** içermeli. Serbest metinli argümanlar (arama `query`'si) için: hangi kelimelerle arandığı değil, hangi KONUDA arandığı ölçülür. `must` içindeki `"*"` yalnızca boş-değil der ve modelin önceki sorgusunu tekrarlamasını da geçirirdi (S04). |
| `must_not_repeat` | `[alt dize]` — çıktıda bu dizelerin **hiçbiri** geçmemeli. Önceki turun cevabının alakasız bir tura taşınmasını yakalar (S07). Kaba ama deterministik; dizeler doğru bir cevapta bulunmayacak şekilde seçilir. |

## Puanlama kuralları

**1. Normalizasyon.** Argüman karşılaştırması küçük harfe indirip baştaki/sondaki
boşlukları atarak yapılır; `İstanbul`/`Istanbul` gibi çiftler zaten `must`
listesinde açıkça sayılmıştır. Sayılar sayısal olarak (`4` = `4.0`), boolean'lar
tip olarak (`true` ≠ `"true"`), diziler küme olarak karşılaştırılır.

**2. Fazladan argüman.** `must` ve `may` dışında bir argüman gelirse hata sayılır.
Şemada olmayan bir argüman uydurulması her zaman hatadır.

**3. Hiçbir kesin alanı ölçülemeyen kayıt "geçti" SAYILMAZ.** Python'da
`all({})` `True` döner; bu tuzağa düşülürse belirsiz bir soruda model tool
çağırdığında kayıt sessizce başarılı görünür ve genel yüzdeyi şişirir. Böyle
kayıtlar paydadan çıkarılır ve "ölçülemedi" olarak ayrıca raporlanır. Bu durum
`boundary_ambiguous` (model tool çağırırsa) ve `I05`/`I06` (model yeniden tool
çağırırsa) kayıtlarında oluşabilir — ikisi de kasıtlı olarak ölçüme kapalı dallar.

**4. `either` beklentisinde rapora sabit değer yazılmaz.** İki davranış da kabul
edilse bile "hep geçti" yazmak sahte bir %100 üretir. Bunun yerine modelin
hangi tarafı seçtiği (`belirsizde_tool_cagirdi`) raporlanır — asıl bilgi budur.

**5. `final_language` YALNIZCA tool çağrısı olmadığında ölçülür.** Model tool
çağırdıysa ortada final turn yoktur; dil ölçümü anlamsızdır ve atlanmalıdır.
Bu, `boundary_ambiguous` ve `missing_argument` kayıtları için özellikle önemli.

**6. `must_not_fabricate_args` nasıl uygulanır:** üretilen her string argüman
değeri, konuşmadaki mesajlarda (normalize edilmiş halde) geçiyor mu diye aranır.
Geçmiyorsa uydurma sayılır. Örnek: "Hava nasıl?" sorusuna
`{"location": "Ankara"}` üretmek hatadır — "Ankara" konuşmada geçmiyor.
Timezone gibi türetilmiş değerler bu kuraldan muaftır (`Seul` → `Asia/Seoul`),
o yüzden kural sadece `missing_argument` kategorisinde uygulanır.

**7. `must_not_fabricate_content` elle kontrol edilir.** Otomatik ölçümü
güvenilir değil; `I05`/`I06` çıktıları gözle okunur. Aranan şey: hata veya boş
sonuç durumunda modelin bilgi uydurup uydurmadığı.

**8. `strict` ve `report` ayrımı.** `strict` alanlar başarı yüzdesine girer.
`report` alanlar tabloda gösterilir ama pass/fail sayılmaz — çünkü o kayıtlarda
tek bir doğru davranış olduğunu iddia edemiyoruz. **Uydurma referans değeri
koymaktansa ölçümü açıkça belirsiz bırakmak tercih edildi.**

**9. Grammar açık/kapalı.** `eval_pre_quant.py` grammar **KAPALI** koşar (modelin
kendi JSON üretme yeteneği ölçülür). `eval_post_quant.py` her iki modda koşar.

## Zorunlu raporlama kırılımı

`eval/README.md` bunu şart koşuyor: sonuçlar **görülen** ve **görülmemiş** tool
şemaları için ayrı ayrı raporlanmalı. Aradaki büyük fark, modelin şema okumayı
değil eğitimdeki tool isimlerini ezberlediğini gösterir.

- görülmemiş şemalı kayıt: **26**
- görülen şemalı kayıt: **60**

`calculate` eğitimde hiç yoktu, dolayısıyla onu içeren her demet `tools_seen: false`
sayılır. `reasoning` (5) ve `compound_request`'in iki kaydı (S01, S02) bu yüzden
"görülmemiş" kovasında — şema genellemesini ölçmedikleri hâlde. Diğer beş
`compound_request` kaydı bilerek üç eğitilmiş tool ile kuruldu ki bu kova
gereksiz şişmesin.

## `compound_request` — tek atışlık eval'in sınırı

Bu kategori, canlıda görülen iki hatadan doğdu: *"hava durumu, sonra 4-5 kaç?"*
sorusunda aritmetik kafadan cevaplandı, *"tensor ve RAG'i ara"* sorusunda RAG hiç
aranmadı ve hiç bahsedilmedi.

Eval **tek atışlıktır**: bir üretim yapar ve onu puanlar, döngüyü koşturmaz. Bu
yüzden "kullanıcının isteği turun SONUNDA tam karşılandı mı" sorusunu doğrudan
ölçemez — model ilk aracı çağırıp ikinciyi bir sonraki iterasyonda çağırabilir ve
bu da geçerli bir stratejidir.

Çözüm, her senaryoyu **iki kayıt** olarak kurmaktır:

- **taze kayıt** (S01, S03, S05) — yalnızca ilk çağrının doğruluğunu ölçer.
  `expected_call_count` burada **rapor**: sıralı strateji de doğrudur, tek doğru
  davranış iddia edilemez (kural 8).
- **devam kaydı** (S02, S04, S06) — ilk gözlem geçmişe **konmuş** hâlde başlar,
  geriye tek bir doğru hamle kalır. Burada **strict** puanlanır. Asıl ölçüm budur.

S07 ayrı durur: önceki turun cevabı geçmişte, soru alakasız. `must_not_repeat` ile
bayat içeriğin sızıp sızmadığı ölçülür.

## Kategoriler

| kategori | n | ölçtüğü |
|---|---|---|
| `seen_positive` | 6 | Referans çizgisi: eğitim dağılımında doğru tool + argüman |
| `seen_negative` | 8 | Tutumluluk: gereksiz tool çağrısı yapmama |
| `boundary_positive` | 6 | Günlük/anlık değişen bilgi → arama şart |
| `boundary_ambiguous` | 4 | İki davranış da savunulabilir (**sadece rapor**) |
| `missing_argument` | 4 | Eksik zorunlu argümanı uydurmama |
| `unseen_positive` | 11 | **Asıl genelleme testi**: hiç görülmemiş şemadan doğru çağrı |
| `unseen_negative` | 5 | Yabancı şemalar varken de tutumlu kalma |
| `wrong_tool_trap` | 5 | Anahtar kelime çakışmasına rağmen doğru seçim |
| `observation_final` | 6 | **Projenin ana hedefi**: gözlem sonrası TÜRKÇE cevap |
| `multi_turn` | 3 | Bağlam taşıma, gereksiz yeniden çağrı yapmama |
| `query_language` | 3 | Türkçe soruya Türkçe arama sorgusu (**rapor**) |
| `multi_call` | 2 | Çoklu tool gereken durumlar (**rapor**) |
| `user_language_edge` | 2 | Kullanıcı İngilizce yazınca ne olur (**rapor**) |
| `compound_request` | 7 | **Tek mesajda birden fazla görev**: hepsi karşılanıyor mu |

`unseen_positive` içindeki 11 kayıt farklı **parametre şekillerini** sınar:
tek zorunlu string, opsiyonel enum, zorunlu integer, hiç zorunlu parametre yok,
boolean, array, üç zorunlu parametre, opsiyonel tarih, şehir/ilçe kırılımı,
kaynak/hedef yönü. Dördü karışık demet kullanır (görülen tool'lar da listede) —
tanıdık isim varken doğru olan yeni tool'u seçebiliyor mu?

## Eğitimle bilinçli örtüşmeler

Sızıntı taraması yapıldı (birebir eşleşme + 0.70 eşiğiyle benzerlik). Kalan beş
örtüşmenin her biri incelendi ve bilerek bırakıldı:

- **A01, A02, A03** (0.73–0.80) — kategori `seen_positive`, yani referans çizgisi.
  Eğitim dağılımına benzemeleri kusur değil, amaç. A03'te şehir (`Seul`) eğitimde
  hiç geçmiyor; sadece cümle kalıbı tanıdık.
- **C01** (0.73) — eğitimde "Apple hissesi şu an kaç dolar?" var ve orada
  `get_stock_price` çağrılıyor. Bizim demette öyle bir tool **yok**; doğru cevap
  `google_search`. Benzerlik testi kolaylaştırmıyor, zorlaştırıyor.
- **E02** (0.80) — "Oradaki saat kaç?" ile "Saat kaç?" kısa cümle olduğu için
  yüksek skor alıyor; niyet farklı (askıda kalan zamir).

Bulunup **düzeltilen** gerçek sızıntılar: `Ankara'da hava nasıl?`,
`Peki İzmir'de?`, `Teşekkürler, çok yardımcı oldun.` (birebir eğitimde vardı),
`New York'ta saat kaç?` (0.91), `Python'da liste ile demet...` (0.75) ve
I05'in ilk hali (eğitimdeki `Ay'da hava nasıl?` → `location_not_found` örneğinin
neredeyse kopyasıydı; görülmemiş tool üzerinden yeniden kuruldu).

## Bakım

Set büyütülürse **her yeni kayıt için** şunlar tekrar koşulmalı:

1. `dogrula_test_set.py` — şema, tip, enum, zorunlu parametre kapsaması, rol
   sırası, `ipython` çift kodlaması, görülmemiş tool isimlerinin gerçekten
   görülmemiş olduğu, birebir sızıntı.
2. Benzerlik taraması (0.70 eşiği) — yakın kopyalar için.
3. `expect`'teki her alanın `scoring`'de karşılığı olmalı. Karşılığı olmayan
   alan sessizce "strict" sayılır; bu belgelenmemiş davranıştır ve E01-E04'te
   bir kez yaşandı.
4. Elle okuma — makine yakalayamaz: referans değerin gerçekten doğru olup
   olmadığı, sorunun tool'un kapsamıyla çelişip çelişmediği (bkz. A05 notu),
   şema açıklamasının testi anlamsızlaştırıp anlamsızlaştırmadığı (bkz. F03 notu).

## Bilinen hoşgörüler

Bilerek gevşek bırakılan, elle bakılması gereken noktalar:

- **F11** — "yarın" için `date` parametresi `may` listesinde ve **değeri
  doğrulanmıyor**; doğru değer koşu tarihine bağlı olurdu. Model yanlış tarih
  yazarsa bu testten kaçar.
- **F04** — `region`/`limit` opsiyonel ve değerleri kontrol edilmiyor; kaydın
  amacı boş `arguments` üretimini ölçmek.
- **B02** — "17 x 23 = 391" gibi salt rakamdan oluşan bir cevapta dil tespiti
  `"?"` döner ve kayıt kalır. Sezgisel yöntemin sınırı; proje zaten doğal
  Türkçe cümle bekliyor.
