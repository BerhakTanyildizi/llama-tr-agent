# =============================================================================
# eval/eval_pre_quant.py
# =============================================================================
# AMAÇ (Spec Bölüm 5.3 + 5.4): LoRA merge edilmiş, HENÜZ KUANTİZE EDİLMEMİŞ
# bfloat16 modelin tool-call başarımını ölçmek. Bu, kuantizasyon sonrası
# ölçümlerin karşılaştırılacağı REFERANS (baseline) değeridir.
#
# BU DOSYANIN YAPACAKLARI:
#   1) Merge edilmiş bfloat16 modeli (transformers ile) yükle.
#   2) eval/test_set.jsonl'i oku — bu set EĞİTİMDE GÖRÜLMEMİŞ tool şemaları içerir.
#   3) Her örnek için modele Llama-3.1 native template ile prompt ver ve
#      tool_call üretmesini iste.
#      ÖNEMLİ: Bu ölçüm GRAMMAR KISITI OLMADAN yapılmalıdır — amaç modelin kendi
#      başına ne kadar geçerli/doğru üretebildiğini görmek. Grammar açıkken JSON
#      geçerliliği zaten %100 çıkar ve ölçüm anlamsızlaşır.
#   4) ÖLÇÜLECEK METRİKLER:
#        a) JSON geçerliliği (parse edilebiliyor mu?)          → SÖZDİZİMSEL
#        b) Doğru tool seçimi (name eşleşiyor mu?)             → SEMANTİK
#        c) Doğru argüman seçimi (parametre adları + değerler) → SEMANTİK
#        d) Gereksiz tool çağrısı / gerekli çağrının atlanması
#        e) Final-turn dili: cevap gerçekten TÜRKÇE mi? (Spec 4.3 doğrulaması)
#   5) GENELLEME TESTİ (Spec 5.3): Sonuçlar, eğitimde görülen tool'lar ile
#      görülmeyen yeni şemalar için AYRI AYRI raporlanır. Aradaki büyük fark,
#      modelin "JSON şema → doğru çağrı" mantığını öğrenmediğini, sadece 10K
#      örnekteki spesifik tool isimlerini ezberlediğini gösterir.
#   6) Sonuçları makine-okunur bir rapora yaz (eval_post_quant.py ile diff için).
# =============================================================================
