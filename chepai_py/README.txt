数据准备过程：
1.运行collectPlate.py 将chepai文件夹中的备选车牌区域进行初步分类
2.手动将T/F_candidate文件夹中的错误图片放入相应的T/F_candidate文件夹中
3.运行changeJpgNames.py 将jpg文件同意更名
4.运行manageData.py 将图片数据整理打包到 *.npz 文件中
5.运行PlateRecogNet.py 使用 *.npz 中的数据对神经网络进行训练，训练结果参数存入PlateRecogNet文件夹中

车牌定位:
运行locatePlate.py 进行车牌定位
如果数据准备过程已经完成，直接运行车牌定位脚本即可


备注：
loadNet.py 只是用于测试训练结果参数是否能被正确导入