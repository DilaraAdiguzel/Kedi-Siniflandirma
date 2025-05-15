import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British', 'Egyptian', 'Maine', 'Persian', 'Ragdoll', 'Russian', 'Siamese', 'Sphynx']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("C:\\Users\\enesa\\OneDrive\\Masaüstü\\\Kedi Siniflandirmasi\\kedi_modeli2.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.set_page_config(layout="wide")

# İlk durumu ayarla
if "started" not in st.session_state:
    st.session_state.started = False
if "show_how_to" not in st.session_state:
    st.session_state.show_how_to = False
if "show_contact" not in st.session_state:
    st.session_state.show_contact = False

# Sağ üst köşeye butonlar
st.markdown("""
    <style>
    .custom-header {
        position: fixed;
        top: 10px;
        right: 20px;
        z-index: 100;
    }
    .custom-header a {
        text-decoration: none;
        color: white;
        background-color: #ff4b4b;
        padding: 8px 16px;
        margin-left: 10px;
        border-radius: 8px;
        font-weight: bold;
        font-family: sans-serif;
        font-size: 14px;
    }
    .custom-header a:hover {
        background-color: #ff7777;
    }
    </style>
""", unsafe_allow_html=True)

# Başka bir durumda
if st.button("Nasıl Çalışır?"):
    st.session_state.show_how_to = not st.session_state.show_how_to
    st.session_state.show_contact = False
elif st.button("İletişim"):
    st.session_state.show_contact = not st.session_state.show_contact
    st.session_state.show_how_to = False

# Nasıl Çalışır? penceresi
if st.session_state.show_how_to:
    with st.expander("", expanded=True):
        st.markdown("""
            <style>
            .info-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .info-header h2 {
                color: #ff4b4b;
            }
            .close-btn {
                background: none;
                border: none;
                font-size: 24px;
                color: #ff4b4b;
                cursor: pointer;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 0.1])
        with col1:
            st.markdown("""
                <div class="info-header">
                    <h2>Kedi Türü Tanıma Sistemi - Nasıl Çalışır?</h2>
                </div>
                <p style='font-size: 16px; color: #333;'>Bu uygulama, yüklediğiniz kedi fotoğrafını analiz ederek hangi tür kedi olduğunu yapay zeka yardımıyla tahmin eder.</p>
                <p style='font-size: 16px; color: #333;'>Eğitim verisi olarak çeşitli kedi ırklarına ait görseller kullanılmıştır ve bu model, <strong>ResNet-18</strong> mimarisi üzerine eğitilmiştir.</p>
                <p style='font-size: 16px; color: #333;'>Yükleme sonrası model, fotoğrafı işler, sınıflandırır ve güven skorunu gösterir.</p>
                <div style='margin-top: 20px; padding: 15px; background-color: #ffedf0; border-radius: 8px;'>
                    <ul style='margin: 0; padding: 0; list-style: none;'>
                        <li style='margin-bottom: 10px;'><span style='font-size: 16px; color: #333;'>📸 Desteklenen formatlar: jpg, jpeg, png</span></li>
                        <li style='margin-bottom: 10px;'><span style='font-size: 16px; color: #333;'>🤖 Yapay zeka modeli: ResNet18 + özel eğitim verisi</span></li>
                        <li style='margin-bottom: 10px;'><span style='font-size: 16px; color: #333;'>🐱 Tanıyabildiği tür sayısı: 12 kedi türü</span></li>
                    </ul>
                </div>
                <p style='font-size: 16px; color: #333; margin-top: 20px;'>Keyifli keşifler!</p>
            """, unsafe_allow_html=True)
        with col2:
            st.button("×", key="close_how_to", on_click=lambda: st.session_state.update(show_how_to=False))

# İletişim penceresi
if st.session_state.show_contact:
    with st.expander("", expanded=True):
        st.markdown("""
            <style>
            .contact-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .contact-header h2 {
                color: #ff4b4b;
            }
            .close-btn {
                background: none;
                border: none;
                font-size: 24px;
                color: #ff4b4b;
                cursor: pointer;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 0.1])
        with col1:
            st.markdown("""
                <div class="contact-header">
                    <h2>İletişim Bilgileri</h2>
                </div>
                <div style='margin-top: 20px;'>
                    <p style='font-size: 16px; color: #333; margin-bottom: 10px;'>E-posta: adiguzeldilara135@gmail.com</p>
                    <p style='font-size: 16px; color: #333;'>Sosyal Medya:</p>
                    <div style='display: flex; gap: 10px; margin-top: 10px;'>
                        <a href='https://www.linkedin.com/in/dilara-ad%C4%B1g%C3%BCzel-046b5831a/' style='color: #333; text-decoration: none;'>Linkedln</a>
                        <a href='https://github.com/DilaraAdiguzel' style='color: #333; text-decoration: none;'>Github</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.button("×", key="close_contact", on_click=lambda: st.session_state.update(show_contact=False))

# Arka plan ve stil
st.markdown("""
    <style>
    .stApp {
        background-color: #ffedf0;
        text-align: center;
    }
    .stImage {
        max-width: 35%;
        max-height: 35%;
    }
    header, footer {
        visibility: hidden;
    }
    .custom-button {
        background-color: #ff4b4b;
        color: white;
        padding: 16px 40px;
        font-size: 20px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 20px;
    }
    .custom-button:hover {
        background-color: #ff7777;
    }

    /* file_uploader için arka plan rengi ve metin rengi */
    section[data-testid="stFileUploader"] {
        background-color: #ff4b4b;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }

    section[data-testid="stFileUploader"] label {
        color: #ff4b4b !important;
        font-weight: bold;
    }

    section[data-testid="stFileUploader"] input[type="file"] {
        color: #ff4b4b !important;
    }

    section[data-testid="stFileUploader"] div {
        color: #ff4b4b !important;
    }
    </style>
""", unsafe_allow_html=True)

# Görsel
image = Image.open("kedyyy.jpeg")
st.image(image, use_container_width=True)

# Buton & durum kontrolü
if not st.session_state.started:
    if st.button("Başla", key="start_button"):
        st.session_state.started = True
        st.rerun()
else:
    # PEMBE BAŞLIK
    st.markdown("""
        <h3 style='color: #ff4b4b; font-weight: bold;'>
            Dosyanızı yükleyin:
        </h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Bir kedi fotoğrafı seçin", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Görseli aç ve dönüştür
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = output.argmax(dim=1)
            confidence = probabilities[0][predicted.item()].item() * 100
            
        # Sayfayı iki sütuna bölmek için
        col1, col2 = st.columns([1, 1])
        
        with col1:  # Sol sütun
            # Görseli göster
            st.image(image, caption='Yüklediğiniz kedi fotoğrafı', width=400)
            
            # Seçilen kedi türüne göre bilgi göster
            cat_info = {
                'Abyssinian': 'Abyssinian kedisi, kökeni oldukça eskiye dayanan ve dünyadaki en eski kedi ırklarından biri olarak bilinen bir kedidir. Bu kedilerin tarihçesi, gizemli ve efsanelerle doludur. Eski Mısır’da kutsal sayılan kedilere olan benzerlikleri, Abyssinian kedilerinin antik çağlardan beri var olduğuna dair birçok teoriyi desteklemektedir.',
                'Bengal': 'Bengal kedileri güzel, zeki ve vahşi görünümlü kedilerdir. Bu melez kedi cinsi, kalıpları ve kişilikleri nedeniyle popülaritesini artırıyor ve büyük bir ev kedisi ile yaklaşık aynı boyutta kalıyor. Bir Asya leopar kedisi (Felis bengalensis - "Bengal" adının türetildiği yer) Abyssinian, Mısır mau veya Amerikan shorthair gibi evcil bir ev kedisiyle üretilerek geliştirildi.',
                'Birman': 'Birman kedisi, ayak ve kulakları beyaz hareli, gözleri mavi bir kedi ırkıdır. Edepli ve duygusal, enerji dolu ve oyuncudur. Birman kedisinin gözleri ve hayranlık uyandıran yüz ifadesi ister evcil olsun, ister vahşi onu diğer kedilerden ayırır.',
                'Bombay': 'Bombay kedisi, Birman kedisi ve siyah American Shorthair kedileri yetiştirilerek, çoğunlukla Birman tipi bir kedi üretmek için geliştirilmiştir. Şık, panter benzeri siyah postlu, kısa tüylü bir kedi türüdür. Bombay kedisine "Siyah Birman" ve "mini panter" de denilmektedir.',
                'British': 'Mısır kökenli olmakla birlikte Büyük Britanya\'yı istila eden Romalılar ile birlikte İngiltere\'ye geldiği düşünülmektedir. Tanınan en eski kedi ırklarından birisidir, yüzyıllar içerisinde çok az değişime uğramıştır. 1914 - 1918 tarihleri arasında Fars Kedisi ile çaprazlanarak uzun tüy genine sahip versiyonu üretilmiştir. Bu şekilde tanıtılan genler, sonunda British Longhair ve Scottish Fold ırklarının temeli haline geldi.',
                'Egyptian': 'Egyptian Mau, dünyadaki en hızlı kedi ırklarından biri olarak bilinir ve saatte 48 kilometre hıza ulaşabilir. Arka bacaklarının ön bacaklarından daha uzun olması, hız ve çeviklik sağlar. Bu kedilerin tüyleri kısa, parlak ve pürüzsüzdür. En dikkat çekici özelliklerinden biri, doğuştan gelen benekli kürk yapısıdır.',
                'Maine': 'Maine Coon, büyük bir kedi ırkıdır. İlk olarak ABD\'de keşfedilmiştir. Bu ırk Kuzey Doğu Amerika bölgesinde keşfedildikten sonra evcilleştirilmiş daha sonra tüm dünyaya yayılmıştır. Dünyanın en büyük evcil kedisi olan Maine Coon ırkının vücut yapısının en gelişmiş haline gelmesi 3 ila 5 yıl arasında değişmektedir.Tüyleri beyaz, siyah, gri, çikolata, leylak gibi renklerde parlak bir görünüme sahiptir.',
                'Persian': 'İran kedisi, yuvarlak yüzü ve ufak ağzıyla ayırt edilen uzun tüylü bir kedi cinsidir. İran üzerinden dünyaya dağılımı Horasan\'dan ihraç edilen kedilerin 1620\'li yıllarda Pietro Della Valle tarafından Ankara üzerinden İtalya\'ya gelmesiyle başlamıştır.',
                'Ragdoll': 'Ragdoll ya da Ragdoll kedisi, iri yapılı, mavi gözlü ve vücudunun belirli bölgelerinde yoğun renk koyulaşmalarına sahip bir kedi ırkıdır. Adını, kucağa alındığında oyuncak bir bez bebek gibi kendisini salıp yayılarak hareketsiz durması sebebiyle "oyuncak bez bebek" anlamına gelen "rag doll" sözcüğünden almıştır.',
                'Russian': 'Russian Blue bir diğer adıyla Rus Mavisi, güzelliği ve tüylerinin parlaklığı ile dikkat çeken kedi türlerinden biridir. Yemyeşil gözleri ile sizi etkisi altına alacak olan Russian Blue cinsi kediler, dışarıdan gelen seslere karşı fazlasıyla duyarlı bir yapıya sahiptirler. Bu nedenle aşırı seslere karşı zaman zaman olumsuz tepkiler verebilen kedilerdir. Çocuklar ile çok iyi anlaşırlar ve oldukça da iyi birer oyuncu kimliğine sahiptirler.',
                'Siamese': 'Siyam kedisi, Güneydoğu Asya\'da resmî adıyla Tayland olarak bilinen Siyam\'da kutsal tapınaklarda beslenen ve buradan tüm dünyaya yayılan bir kedi ırkıdır. Siyam kedisi, Güneydoğu Asya\'da görülen bir cins kedi olduğu için ismini Siyam\'dan alır. Tayland dilindeki adı Wichien Maat\'tır.',
                'Sphynx': 'Sphynx, Dünyadaki birkaç tüysüz kedi cinsinden biridir. Kökeni Kanada\'dır. Görüntüsünün aksine yumuşak huylu, sahibine bağlı ve cana yakın bir türdür. Ömürleri 8 ila 14 yıl arasındadır. Tüyleri olmadığından soğuğa karşı dayanıksızdır. Bulunduğu ortam sıcak olmalıdır, bunun yanı sıra giysiler giydirerek de bu denge sağlanabilir.'
            }
            
            # Seçilen kedi türüne göre bilgi göster
            cat_type = class_names[predicted.item()]
            st.markdown(f"""
                <div style='background-color: #ffedf0; padding: 20px; border-radius: 10px; text-align: center; width: 100%;'>
                    <h4 style='color: #ff4b4b; margin-bottom: 15px;'>{cat_type} Kedisi Hakkında</h4>
                    <p style='font-size: 14px; color: #333;'>{cat_info[cat_type]}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:  # Sağ sütun
            # Sonuçları göster
            st.markdown(f"""
                <h3 style='color: #ff4b4b; font-weight: bold; margin-top: 20px;'>Tahmin Sonucu:</h3>
                <div style='display: flex; flex-direction: column; gap: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <span style='font-size: 24px; color: #333;'>Bu kedi türü: </span>
                        <span style='font-size: 24px; color: #ff4b4b; font-weight: bold;'>{class_names[predicted.item()]}</span>
                    </div>
                    <div style='display: flex; align-items: center;'>
                        <span style='font-size: 18px; color: #666;'>Tahmin doğruluk oranı: </span>
                        <span style='font-size: 18px; color: #ff4b4b; font-weight: bold;'>{confidence:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            