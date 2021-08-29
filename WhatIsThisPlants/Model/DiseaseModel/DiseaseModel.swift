//
//  DiseaseModel.swift
//  WhatIsThisPlants
//
//  Created by Seonghun Kim on 2021/08/29.
//

import Foundation
import TFLiteSwift_Vision

final class DiseaseModel: ImageInferable {
    private enum Resource {
        static let modelName = "lite-model_disease-classification_1"
    }
    
    private lazy var labels: [String] = [
        "Tomato Healthy",
        "Tomato Septoria Leaf Spot",
        "Tomato Bacterial Spot",
        "Tomato Blight",
        "Cabbage Healthy",
        "Tomato Spider Mite",
        "Tomato Leaf Mold",
        "Tomato_Yellow Leaf Curl Virus",
        "Soy_Frogeye_Leaf_Spot",
        "Soy_Downy_Mildew",
        "Maize_Ravi_Corn_Rust",
        "Maize_Healthy",
        "Maize_Grey_Leaf_Spot",
        "Maize_Lethal_Necrosis",
        "Soy_Healthy",
        "Cabbage Black Rot"
    ]
    
    
    private lazy var visionInterpreter: TFLiteVisionInterpreter = {
        let interpreterOptions = TFLiteVisionInterpreter.Options(
            modelName: Resource.modelName,
            normalization: .meanStd(mean: [127.5], std: [127.5])
        )
        return TFLiteVisionInterpreter(options: interpreterOptions)
    }()
    
    func process(pixelBuffer: CVPixelBuffer) -> Inference {
        let input: TFLiteVisionInput = .pixelBuffer(pixelBuffer: pixelBuffer)
        
        guard let inputData = visionInterpreter.preprocess(with: input) else {
            fatalError("Failure to preprcess")
        }

        guard let outputs = visionInterpreter.inference(with: inputData)?.first else {
            fatalError("Failure to  inference")
        }

        let predictedIndex = Int(outputs.argmax())
        
        let predictedThreshold = outputs.array[predictedIndex]
        let predictedLabel = self.labels[predictedIndex]

        return (label: predictedLabel, threshold: predictedThreshold)
    }
}

// https://tfhub.dev/agripredict/disease-classification/1
