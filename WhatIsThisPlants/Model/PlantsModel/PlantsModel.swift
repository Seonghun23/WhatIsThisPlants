//
//  PlantsModel.swift
//  WhatIsThisPlants
//
//  Created by Seonghun Kim on 2021/08/29.
//

import Foundation
import TFLiteSwift_Vision
import CoreMedia
import CoreVideo

final class PlantsModel: ImageInferable {
    private enum Resource {
        static let modelName = "lite-model_aiy_vision_classifier_plants_V1_3"
        static let labelFileName = "aiy_plants_V1_labelmap"
    }
    
    private lazy var labels: [Int: String] = {
        guard let labelFilePath = Bundle.main.path(forResource: Resource.labelFileName, ofType: "csv") else {
            assertionFailure("Failure get labels::\(Resource.labelFileName)")
            return [:]
        }
        var labels: [Int: String] = [:]
        do {
            try String(contentsOfFile: labelFilePath, encoding: .utf8)
                .components(separatedBy: "\n")
                .forEach({ text in
                    let labelInfo = text.components(separatedBy: ",")
                    
                    guard let index = text.first.map(String.init).flatMap(Int.init) else { return }
                    guard let label = text.last.map(String.init) else { return }
                
                    labels[index] = label
                })
        } catch let error {
            assertionFailure(error.localizedDescription)
        }
        return labels
    }()
    
    
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
        
        guard let predictedLabel = self.labels[predictedIndex] else {
            fatalError("labels doesn't contain index label")
        }

        return (label: predictedLabel, threshold: predictedThreshold)
    }
}

// https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1
