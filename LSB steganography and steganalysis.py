from Common.dct2d import dct, idct
import numpy as np
import cv2
import matplotlib.pyplot as plt
#이미지 불러와서 imageB의 크기에 맞게 resize
imageB = cv2.imread("./B.jpg", cv2.IMREAD_GRAYSCALE) #570, 380

imageA = cv2.imread("./A.jpg", cv2.IMREAD_COLOR)
imageA = cv2.resize(imageA, (570, 380))
imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
image1 = cv2.imread("./1.jpg", cv2.COLOR_BGR2RGB)
image1 = cv2.resize(image1, (570, 380))
image2 = cv2.imread("./2.jpg", cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image2, (570, 380))
image3 = cv2.imread("./3.png", cv2.COLOR_BGR2RGB) #편향된 데이터
image3 = cv2.resize(image3, (570, 380))
image4 = cv2.imread("./4.jpg", cv2.COLOR_BGR2RGB)
image4 = cv2.resize(image4, (570, 380))
image6 = cv2.imread("./6.jpg", cv2.COLOR_BGR2RGB)
image6 = cv2.resize(image6, (570, 380))

w, h = imageB.shape[:2]

_, imageB_gray = cv2.threshold(imageB, 115.5, 255, cv2.THRESH_BINARY) #흑백 픽셀 약 100개정도 차이
imageB_gray = cv2.normalize(imageB_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)#255값을 1로 변환

#블루채널 꺼내오기
def LSBPutImage(imageA,imageB):
    #블루채널만 복사
    cBlue = imageA[:, :, 2].copy()
    
    #LSB에 image B 기록
    for i in range(w):
        for j in range(h):
            if imageB[i, j] == 0: #imageB_gray pixel값이 0일때
                if cBlue[i, j] % 2 != 0: # LSB가 홀수일때
                    cBlue[i, j] -= 1
            else: #imageB_gray pixel값이 1일때
                if cBlue[i, j] % 2 == 0:# LSB가 짝수일때
                    cBlue[i, j] += 1

    imageAputB = imageA.copy()
    imageAputB[:, :, 2] = cBlue
    return imageAputB

#imageA의 LSB추출해서 imageB 재구성
def LSB(imageAputB):
    reviseLSB = imageAputB[:, :, 2]
    reviseLSB = np.where(reviseLSB % 2 == 0, 0, 1)
    return reviseLSB


imageAputB = LSBPutImage(imageA, imageB_gray)
image1putB = LSBPutImage(image1, imageB_gray)
image2putB = LSBPutImage(image2, imageB_gray)
image3putB = LSBPutImage(image3, imageB_gray)
image4putB = LSBPutImage(image4, imageB_gray)
image6putB = LSBPutImage(image6, imageB_gray)


# #재구성한 imageB
# imageAputB_LSB = LSB(imageAputB)
# plt.imshow(imageAputB_LSB, 'gray')
# plt.show()

# # imageA와 imageA'의 LSB값 이진이미지로 출력
# imageA_LSB=LSB(imageA)
# plt.imshow(imageA_LSB, 'gray')
# plt.show()
# plt.imshow(imageAputB_LSB, 'gray')
# plt.show()

# # imageA와 imageA' 비교
# plt.imshow(imageA)
# plt.show()
# plt.imshow(imageAputB)
# plt.show()

def cos_sim(a, b): #유사도 측정에 사용
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def detect(image): #LSB embedding 이미지라면 1, 아니면 0을 리턴
    result=[]
    for j in range(3):
        n, _, _ = plt.hist(image[:,:,j].flatten(), 256, [0, 256]) #픽셀값에 해당하는 빈도수 저장
        even = []
        odd = []
        for i in range(256):
            if i == 0 or i == 255: #맨처음과 끝값은 인접값의 중간값으로 넣을수 없기때문에 기존의 값 넣어줌
                even.append(n[i])
                odd.append(n[i])
            else:
                if i % 2 == 0: #짝수 픽셀의 빈도수일 경우
                    even.append(n[i])
                    odd.append(np.median([n[i - 1],n[i + 1]])) #인접값의 중간값으로 채우기
                else:#홀수 픽셀의 빈도수일 경우
                    even.append(np.median([n[i - 1],n[i + 1]]))
                    odd.append(n[i])
        result.append(cos_sim(even, odd))
    result=np.array(result)
    result=np.where(result>=0.99, 0, 1) #유사도가 0.99이상이면 원본이미지, 미만이면 임베딩된 이미지
    return max(result) #하나라도 1인경우 1을 return하게 됨

# binary classifier 구현 테스트
print("=====원본 이미지=====")
print(detect(imageA))
print(detect(image1))
print(detect(image2))
print(detect(image3),'(편향된이미지)')
print(detect(image4))
print(detect(image6))
print("=====LSB embedding 이미지=====")
print(detect(imageAputB))
print(detect(image1putB))
print(detect(image2putB))
print(detect(image3putB),'(편향된이미지)')
print(detect(image4putB))
print(detect(image6putB))

