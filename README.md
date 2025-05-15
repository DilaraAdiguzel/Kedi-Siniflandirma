# Kedi Türü Sınıflandırma Uygulaması

Bu proje, Oxford-IIIT Pet Dataset kullanılarak eğitilen bir **Convolutional Neural Network (CNN)** modeli ile kedi türlerini sınıflandıran bir **Streamlit** tabanlı web uygulamasıdır.

## İçindekiler

- [Genel Bakış](#genel-bakış)  
- [Veri Seti](#veri-seti)  
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)  
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Uygulama Kullanımı](#uygulama-kullanımı)  
- [Model Bilgisi](#model-bilgisi)  
 
- [İletişim](#iletişim)

---

## Genel Bakış

Bu proje, kullanıcıların yükledikleri kedi fotoğraflarına göre hangi türden olduğunu tahmin eden bir sınıflandırma sistemidir. Uygulama, eğitilmiş bir yapay zeka modeli ve sade bir kullanıcı arayüzü sunar. Kullanım kolaylığı göz önünde bulundurularak tasarlanmıştır.

---

## Veri Seti

- **Oxford-IIIT Pet Dataset** kullanılmıştır.  
- Toplamda 2000'den fazla **etiketli kedi görseli** içerir.  
- Görseller farklı ırk ve özelliklere sahip kedilere aittir. (12 ırk) 
- Resmi bağlantı: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

---

## Kullanılan Teknolojiler

- Python 3.11
- Streamlit  
- Pytorch 
- NumPy, Pandas ,PIL


---

## Kurulum ve Çalıştırma

1. **Repository’yi klonlayın:**

```bash
git clone https://github.com/kullaniciadi/kedi-siniflandirma.git
cd kedi-siniflandirma



Uygulamayı başlatın:

streamlit run app.py


##Uygulama Kullanımı
Arayüz üzerinden bir kedi fotoğrafı yükleyin.

Sistem, fotoğrafı işleyip önceden eğitilmiş modelle sınıflandırma yapar.

Sonuç, arayüzde tahmin edilen kedi türü ve o kedi ırkına ait bir genel bilgilendirme olarak gösterilir.

Uygulama aynı zamanda yüklenen görseli ve modelin verdiği güven oranını da (accuracy/probability) görüntüler.


##İletişim
Proje sahibiyle iletişime geçmek için:

Ad Soyad: Dilara Adıgüzel

E-posta: adiguzeldilara135@gmail.com

GitHub: https://github.com/DilaraAdiguzel


