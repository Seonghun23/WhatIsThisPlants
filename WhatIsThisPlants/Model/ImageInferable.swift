//
//  ImageInferable.swift
//  WhatIsThisPlants
//
//  Created by Seonghun Kim on 2021/08/29.
//

import Foundation
import TFLiteSwift_Vision

protocol ImageInferable {
    func process(pixelBuffer: CVPixelBuffer) -> Inference
}

extension ImageInferable {
    typealias Inference = (label: String, threshold: Float32)
}
