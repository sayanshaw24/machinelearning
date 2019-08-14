// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.Summaries;
using static Microsoft.ML.Transforms.Dnn.DnnUtils;
using static Microsoft.ML.Transforms.DnnEstimator;
using static Tensorflow.Python;

[assembly: LoadableClass(DnnTransformer.Summary, typeof(IDataTransform), typeof(DnnTransformer),
    typeof(DnnEstimator.Options), typeof(SignatureDataTransform), DnnTransformer.UserName, DnnTransformer.ShortName)]

[assembly: LoadableClass(DnnTransformer.Summary, typeof(IDataTransform), typeof(DnnTransformer), null, typeof(SignatureLoadDataTransform),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(DnnTransformer), null, typeof(SignatureLoadModel),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DnnTransformer), null, typeof(SignatureLoadRowMapper),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer" /> for the <see cref="DnnEstimator"/>.
    /// </summary>
    public sealed class DnnTransformer : RowToRowTransformerBase
    {
        private readonly IHostEnvironment _env;
        private readonly string _modelLocation;
        private readonly bool _transferLearning;
        private readonly bool _isTemporarySavedModel;
        private readonly bool _addBatchDimensionInput;
        private Session _session;
        private readonly DataViewType[] _outputTypes;
        private readonly TF_DataType[] _tfOutputTypes;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly TensorShape[] _tfInputShapes;
        private readonly (Operation, int)[] _tfInputOperations;
        private readonly (Operation, int)[] _tfOutputOperations;
        private TF_Output[] _tfInputNodes;
        private readonly TF_Output[] _tfOutputNodes;
        private Tensor _bottleneckTensor;
        private Operation _trainStep;
        private Tensor _softMaxTensor;
        private Tensor _crossEntropy;
        private Tensor _labelTensor;
        private Tensor _evaluationStep;
        private Tensor _prediction;
        private readonly int _classCount;
        private readonly string _checkpointPath;
        private readonly string _bottleneckOperationName;
        private Graph Graph => _session.graph;
        private readonly Dictionary<string, string> _idvToTfMapping;
        private readonly string[] _inputs;
        private readonly string[] _outputs;
        private readonly string _labelColumnName;
        private readonly string _checkpointName;
        private readonly Architecture _arch;
        private readonly string _scoreColumnName;
        private readonly string _predictedLabelColumnName;
        private readonly float _learningRate;
        private readonly string _softmaxTensorName;
        private readonly string _predictionTensorName;

        internal const string Summary = "Trains Dnn models.";
        internal const string UserName = "DnnTransform";
        internal const string ShortName = "DnnTransform";
        internal const string LoaderSignature = "DnnTransform";

        internal static class DefaultModelFileNames
        {
            public const string VariablesFolder = "variables";
            public const string Index = "variables.index";
            public const string Data = "variables.data-00000-of-00001";
            public const string Graph = "saved_model.pb";
            public const string TmpMlnetModel = "mlnet_model";
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DNNTRANS",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00000001,
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DnnTransformer).Assembly.FullName);
        }

        // Factory method for SignatureLoadModel.
        private static DnnTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            // stream: tensorFlow model.

            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput,
                out bool transferLearning, out string labelColumn, out string checkpointName, out Architecture arch, out string scoreColumnName,
                out string predictedColumnName, out float learningRate, out int classCount, out string predictionTensorName, out string softMaxTensorName);

            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();

                return new DnnTransformer(env, DnnUtils.LoadTFSession(env, modelBytes), outputs, inputs,
                    null, false, addBatchDimensionInput, 1, transferLearning, labelColumn, checkpointName, arch,
                    scoreColumnName, predictedColumnName, learningRate, null, classCount, true, predictionTensorName, softMaxTensorName);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(DnnTransformer) + "_" + Guid.NewGuid()));
            DnnUtils.CreateFolderWithAclIfNotExists(env, tempDirPath);
            try
            {
                var load = ctx.TryLoadBinaryStream("TFSavedModel", br =>
                {
                    int count = br.ReadInt32();
                    for (int n = 0; n < count; n++)
                    {
                        string relativeFile = br.ReadString();
                        long fileLength = br.ReadInt64();

                        string fullFilePath = Path.Combine(tempDirPath, relativeFile);
                        string fullFileDir = Path.GetDirectoryName(fullFilePath);
                        if (fullFileDir != tempDirPath)
                        {
                            DnnUtils.CreateFolderWithAclIfNotExists(env, fullFileDir);
                        }
                        using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                        {
                            long actualRead = br.BaseStream.CopyRange(fs, fileLength);
                            env.Assert(actualRead == fileLength);
                        }
                    }
                });

                return new DnnTransformer(env, DnnUtils.GetSession(env, tempDirPath), outputs, inputs, tempDirPath, true,
                    addBatchDimensionInput, 1, transferLearning, labelColumn, checkpointName, arch,
                    scoreColumnName, predictedColumnName, learningRate, null, classCount, true, predictionTensorName, softMaxTensorName);
            }
            catch (Exception)
            {
                DnnUtils.DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, DnnEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new DnnTransformer(env, options, input).MakeDataTransform(input);
        }

        internal DnnTransformer(IHostEnvironment env, DnnEstimator.Options options, IDataView input)
            : this(env, options, DnnUtils.LoadDnnModel(env, options.ModelLocation), input)
        {
        }

        internal DnnTransformer(IHostEnvironment env, DnnEstimator.Options options, DnnModel tensorFlowModel, IDataView input, IDataView validationSet = null)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns,
                  options.ModelLocation, false, options.AddBatchDimensionInputs, options.BatchSize, options.TransferLearning,
                  options.LabelColumn, options.CheckpointName, options.Arch, options.ScoreColumnName,
                  options.PredictedLabelColumnName, options.LearningRate, input.Schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            if (options.ReTrain)
                CheckTrainingParameters(options);

            if (options.ReTrain && !DnnUtils.IsSavedModel(env, options.ModelLocation))
                throw env.ExceptNotSupp("TensorFlowTransform: Re-Training of TensorFlow model is only supported for un-frozen model.");

            TrainCore(options, input, validationSet);
        }

        private void CheckTrainingParameters(DnnEstimator.Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            Host.CheckNonWhiteSpace(options.OptimizationOperation, nameof(options.OptimizationOperation));
            if (_session.graph.OperationByName(options.OptimizationOperation) == null)
                throw Host.ExceptParam(nameof(options.OptimizationOperation), $"Optimization operation '{options.OptimizationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.TensorFlowLabel, nameof(options.TensorFlowLabel));
            if (_session.graph.OperationByName(options.TensorFlowLabel) == null)
                throw Host.ExceptParam(nameof(options.TensorFlowLabel), $"'{options.TensorFlowLabel}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveLocationOperation, nameof(options.SaveLocationOperation));
            if (_session.graph.OperationByName(options.SaveLocationOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveLocationOperation), $"'{options.SaveLocationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveOperation, nameof(options.SaveOperation));
            if (_session.graph.OperationByName(options.SaveOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveOperation), $"'{options.SaveOperation}' does not exist in the model");

            if (options.LossOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LossOperation, nameof(options.LossOperation));
                if (_session.graph.OperationByName(options.LossOperation) == null)
                    throw Host.ExceptParam(nameof(options.LossOperation), $"'{options.LossOperation}' does not exist in the model");
            }

            if (options.MetricOperation != null)
            {
                Host.CheckNonWhiteSpace(options.MetricOperation, nameof(options.MetricOperation));
                if (_session.graph.OperationByName(options.MetricOperation) == null)
                    throw Host.ExceptParam(nameof(options.MetricOperation), $"'{options.MetricOperation}' does not exist in the model");
            }

            if (options.LearningRateOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LearningRateOperation, nameof(options.LearningRateOperation));
                if (_session.graph.OperationByName(options.LearningRateOperation) == null)
                    throw Host.ExceptParam(nameof(options.LearningRateOperation), $"'{options.LearningRateOperation}' does not exist in the model");
            }
        }

        private (int, bool, TF_DataType, TensorShape) GetTrainingInputInfo(DataViewSchema inputSchema, string columnName, string tfNodeName, int batchSize)
        {
            if (!inputSchema.TryGetColumnIndex(columnName, out int inputColIndex))
                throw Host.Except($"Column {columnName} doesn't exist");

            var type = inputSchema[inputColIndex].Type;
            var isInputVector = type is VectorDataViewType;

            (Operation inputTensor, int index) = GetOperationFromName(tfNodeName, _session);
            var tfInput = new TF_Input(inputTensor, index);
            var tfInputType = inputTensor.OpType == "Placeholder" ? inputTensor.OutputType(index) :
                inputTensor.InputType(index);
            var tfInputShape = ((Tensor)inputTensor).TensorShape;

            if (isInputVector && (tfInputShape == null || (tfInputShape.NDim == 0)))
            {
                var vecType = (VectorDataViewType)type;
                var colTypeDims = new int[vecType.Dimensions.Length + 1];
                colTypeDims[0] = -1;
                for (int indexLocal = 0; indexLocal < vecType.Dimensions.Length; indexLocal += 1)
                    colTypeDims[indexLocal + 1] = vecType.Dimensions[indexLocal];

                tfInputShape = new TensorShape(colTypeDims);
            }
            if (tfInputShape.NDim != -1)
            {
                var newShape = new int[tfInputShape.NDim];
                newShape[0] = tfInputShape[0] == 0 || tfInputShape[0] == -1 ? batchSize : tfInputShape[0];

                for (int j = 1; j < tfInputShape.NDim; j++)
                    newShape[j] = tfInputShape[j];
                tfInputShape = new TensorShape(newShape);
            }

            var expectedType = DnnUtils.Tf2MlNetType(tfInputType);
            var actualType = type.GetItemType().RawType;
            if (type is KeyDataViewType && actualType == typeof(UInt32))
                actualType = typeof(Int64);

            if (actualType != expectedType.RawType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", columnName, expectedType.ToString(), type.ToString());

            return (inputColIndex, isInputVector, tfInputType, tfInputShape);
        }

        private void TrainCore(DnnEstimator.Options options, IDataView input, IDataView validationSet)
        {
            var inputsForTraining = new string[_inputs.Length + 1];
            var inputColIndices = new int[inputsForTraining.Length];
            var isInputVector = new bool[inputsForTraining.Length];
            var tfInputTypes = new TF_DataType[inputsForTraining.Length];
            var tfInputShapes = new TensorShape[inputsForTraining.Length];

            for (int i = 0; i < _inputs.Length; i++)
                inputsForTraining[i] = _idvToTfMapping[_inputs[i]];

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length - 1; i++)
                (inputColIndices[i], isInputVector[i], tfInputTypes[i], tfInputShapes[i]) =
                    GetTrainingInputInfo(inputSchema, _inputs[i], inputsForTraining[i], options.BatchSize);

            var index = inputsForTraining.Length - 1;
            if (options.TransferLearning)
                inputsForTraining[index] = _labelTensor.name.Split(':').First();
            else
                inputsForTraining[index] = options.TensorFlowLabel;

            (inputColIndices[index], isInputVector[index], tfInputTypes[index], tfInputShapes[index]) =
                    GetTrainingInputInfo(inputSchema, options.LabelColumn, inputsForTraining[index], options.BatchSize);

            // Create graph inputs.
            Operation labelOp;
            int labelOpIdx;
            if (options.ReTrain)
                (labelOp, labelOpIdx) = GetOperationFromName(options.TensorFlowLabel, _session);
            else
                (labelOp, labelOpIdx) = GetOperationFromName(_labelTensor.name, _session);

            TF_Output[] tfInputs;

            if (options.ReTrain && !string.IsNullOrEmpty(options.LearningRateOperation))
                tfInputs = new TF_Output[_tfInputNodes.Length + 2]; //Inputs + Label + Learning Rate.
            else
                tfInputs = new TF_Output[_tfInputNodes.Length + 1]; //Inputs + Label.

            Array.Copy(_tfInputNodes, tfInputs, _tfInputNodes.Length);

            tfInputs[_tfInputNodes.Length] = new TF_Output(labelOp, labelOpIdx);

            if (options.ReTrain)
            {
                var lr = GetOperationFromName(options.LearningRateOperation, _session);
                tfInputs[_tfInputNodes.Length + 1] = new TF_Output(lr.Item1, lr.Item2);
            }

            // Create graph operations.
            IntPtr[] ops = null;
            if (options.ReTrain && options.OptimizationOperation != null)
                ops = new[] { c_api.TF_GraphOperationByName(Graph, options.OptimizationOperation) };
            else
                ops = new[] { (IntPtr)_trainStep };

            Saver trainSaver = null;
            FileWriter trainWriter = null;
            Tensor merged = null;
            Runner testSetRunner = null;
            Runner validationSetRunner = null;
            if (options.TransferLearning)
            {
                merged = tf.summary.merge_all();
                trainWriter = tf.summary.FileWriter(Path.Combine(Directory.GetCurrentDirectory(), "train"), _session.graph);
                trainSaver = tf.train.Saver();
                trainSaver.save(_session, _checkpointPath);
            }

            // Instantiate the graph.
            Runner runner;
            var cols = input.Schema.Where(c => inputColIndices.Contains(c.Index));

            for (int epoch = 0; epoch < options.Epoch; epoch++)
            {
                using (var cursor = input.GetRowCursor(cols))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);
                    bool isDataLeft = false;
                    using (var ch = Host.Start("Training TensorFlow model..."))
                    using (var pch = Host.StartProgressChannel("TensorFlow training progress..."))
                    {
                        if (options.ReTrain)
                        {
                            float loss = 0;
                            float metric = 0;
                            pch.SetHeader(new ProgressHeader(new[] { "Loss", "Metric" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputsForTraining.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    runner = new Runner(_session);

                                    // Add Learning Rate.
                                    if (!string.IsNullOrEmpty(options.LearningRateOperation))
                                        runner.AddInput(options.LearningRateOperation, new Tensor(options.LearningRate));

                                    // Add operations.
                                    if (!string.IsNullOrEmpty(options.OptimizationOperation))
                                        runner.AddOperation(options.OptimizationOperation);

                                    // Add outputs.
                                    if (options.LossOperation != null)
                                        runner.AddOutputs(options.LossOperation);
                                    if (options.MetricOperation != null)
                                        runner.AddOutputs(options.MetricOperation);

                                    var (l, m) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, runner);
                                    loss += l;
                                    metric += m;
                                }
                            }
                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                            pch.Checkpoint(new double?[] { loss, metric });
                        }
                        else
                        {
                            pch.SetHeader(new ProgressHeader(null, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputsForTraining.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    runner = new Runner(_session);

                                    // Add operations.
                                    runner.AddOperation(_trainStep);

                                    // Feed inputs.
                                    for (int i = 0; i < inputsForTraining.Length; i++)
                                        runner.AddInput(inputsForTraining[i], srcTensorGetters[i].GetBufferedBatchTensor());

                                    // Execute the graph.
                                    var t = runner.Run();
                                }
                            }

                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                        }
                    }
                }

                // Measure accuracy of the model.
                if (options.TransferLearning && options.MeasureTrainAccuracy)
                {
                    // Test on the training set to get accuracy.
                    using (var cursor = input.GetRowCursor(cols))
                    {
                        var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                        float accuracy = 0;
                        float crossEntropy = 0;
                        bool isDataLeft = false;
                        int batch = 0;
                        using (var ch = Host.Start("Test TensorFlow model..."))
                        using (var pch = Host.StartProgressChannel("TensorFlow testing progress..."))
                        {
                            pch.SetHeader(new ProgressHeader(new[] { "Accuracy", "Cross Entropy" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputColIndices.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    testSetRunner = new Runner(_session);
                                    testSetRunner.AddOutputs(_evaluationStep.name);
                                    testSetRunner.AddOutputs(_crossEntropy.name);
                                    testSetRunner.AddOutputs(_bottleneckTensor.name);
                                    var (acc, ce) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, testSetRunner);
                                    accuracy += acc;
                                    crossEntropy += ce;
                                    batch++;
                                }
                            }

                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                            pch.Checkpoint(new double?[] { accuracy / batch, crossEntropy / batch });
                            ch.Info(MessageSensitivity.None, $"Accuracy: {accuracy / batch}, Cross-Entropy: {crossEntropy / batch}");
                        }
                    }

                    // Test on the validation set.
                    if (validationSet != null)
                    {
                        using (var cursor = validationSet.GetRowCursor(cols))
                        {
                            var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                            float accuracy = 0;
                            bool isDataLeft = false;
                            int batch = 0;
                            using (var ch = Host.Start("Test TensorFlow model with validation set..."))
                            using (var pch = Host.StartProgressChannel("TensorFlow validation progress..."))
                            {
                                pch.SetHeader(new ProgressHeader(new[] { "Accuracy" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                                while (cursor.MoveNext())
                                {
                                    for (int i = 0; i < inputColIndices.Length; i++)
                                    {
                                        isDataLeft = true;
                                        srcTensorGetters[i].BufferTrainingData();
                                    }

                                    if (((cursor.Position + 1) % options.BatchSize) == 0)
                                    {
                                        isDataLeft = false;
                                        validationSetRunner = new Runner(_session);
                                        validationSetRunner.AddOutputs(_evaluationStep.name);
                                        var (acc, _) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, validationSetRunner);
                                        accuracy += acc;
                                        batch++;
                                    }
                                }
                                if (isDataLeft)
                                {
                                    isDataLeft = false;
                                    ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                                }
                                pch.Checkpoint(new double?[] { accuracy / batch });
                            }
                        }
                    }
                }
            }

            if (options.ReTrain)
                UpdateModelOnDisk(options.ModelLocation, options);
            else
            {
                trainSaver.save(_session, _checkpointPath);
                UpdateTransferLearningModelOnDisk(options, _classCount);
            }
        }

        private (float loss, float metric) ExecuteGraphAndRetrieveMetrics(
            string[] inputs,
            ITensorValueGetter[] srcTensorGetters,
            Runner runner)
        {
            float loss = 0;
            float metric = 0;
            for (int i = 0; i < inputs.Length; i++)
                runner.AddInput(inputs[i], srcTensorGetters[i].GetBufferedBatchTensor());

            Tensor[] tensor = runner.Run();
            var buffer = tensor[0].Data();
            loss = tensor.Length > 0 && tensor[0] != IntPtr.Zero ? (float)tensor[0].Data<float>()[0] : 0.0f;
            metric = tensor.Length > 1 && tensor[1] != IntPtr.Zero ? (float)tensor[1].Data<float>()[0] : 0.0f;
            var b = tensor.Length > 2 && tensor[2] != IntPtr.Zero ? (float[])tensor[2].Data<float>() : null;
            return (loss, metric);
        }

        /// <summary>
        /// Updates the model on the disk.
        /// After retraining Session and Graphs are both up-to-date
        /// However model on disk is not which is used to serialzed to ML.Net stream
        /// </summary>
        private void UpdateModelOnDisk(string modelDir, DnnEstimator.Options options)
        {
            try
            {
                // Save the model on disk
                var path = Path.Combine(modelDir, DefaultModelFileNames.TmpMlnetModel);
                //var input = GetOperationFromName(options.SaveLocationOperation, Session);
                var runner = new Runner(_session); //, new[] { new TF_Output(input.Item1, input.Item2) }, null, new[] { c_api.TF_GraphOperationByName(Graph, options.SaveOperation) });

                runner.AddInput(options.SaveLocationOperation, new Tensor(path))
                    .AddOperation(options.SaveOperation)
                    .Run();

                // Preserve original files
                var variablesPath = Path.Combine(modelDir, DefaultModelFileNames.VariablesFolder);
                var archivePath = Path.Combine(variablesPath + "-" + Guid.NewGuid().ToString());
                Directory.CreateDirectory(archivePath);
                foreach (var f in Directory.GetFiles(variablesPath))
                    File.Copy(f, Path.Combine(archivePath, Path.GetFileName(f)));

                string[] modelFilePaths = null;

                // There are two ways parameters are saved depending on
                // either `saver_def = tf.train.Saver().as_saver_def()` was called in Python before `tf.saved_model.simple_save` or not.
                // If `saver_def = tf.train.Saver().as_saver_def()` was called files are saved in top directory.
                // If not then temporary directory is created in current directory which starts with `mlnet_model`
                // and files are saved there.
                var tmpParamDir = Directory.GetDirectories(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");
                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    modelFilePaths = Directory.GetFiles(tmpParamDir[0]);
                else
                    modelFilePaths = Directory.GetFiles(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");

                foreach (var file in modelFilePaths)
                {
                    if (file.EndsWith(".data-00000-of-00001"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Data);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                    if (file.EndsWith(".index"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Index);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                }

                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    DnnUtils.DeleteFolderWithRetries(Host, tmpParamDir[0]);
            }
            catch (Exception e)
            {
                throw Host.ExceptIO(e, "Error serializing TensorFlow retrained model to disk.");
            }
        }

        private (Session, Tensor, Tensor, Tensor) BuildEvaluationSession(DnnEstimator.Options options, int classCount)
        {
            var evalGraph = DnnUtils.LoadMetaGraph(options.ModelLocation);
            var evalSess = tf.Session(graph: evalGraph);
            Tensor evaluationStep = null;
            Tensor prediction = null;
            Tensor bottleneckTensor = evalGraph.OperationByName(_bottleneckOperationName);

            tf_with(evalGraph.as_default(), graph =>
            {
                var (_, _, groundTruthInput, finalTensor) = AddFinalRetrainOps(classCount, options.LabelColumn,
                    options.ScoreColumnName, options.LearningRate, bottleneckTensor, false);

                tf.train.Saver().restore(evalSess, Path.Combine(Directory.GetCurrentDirectory(), _checkpointPath));
                (evaluationStep, prediction) = AddEvaluationStep(finalTensor, groundTruthInput);
            });

            return (evalSess, _labelTensor, evaluationStep, prediction);
        }

        private (Tensor, Tensor) AddEvaluationStep(Tensor resultTensor, Tensor groundTruthTensor)
        {
            Tensor evaluationStep = null;
            Tensor correctPrediction = null;

            tf_with(tf.name_scope("accuracy"), scope =>
            {
                tf_with(tf.name_scope("correct_prediction"), delegate
                {
                    _prediction = tf.argmax(resultTensor, 1);
                    correctPrediction = tf.equal(_prediction, groundTruthTensor);
                });

                tf_with(tf.name_scope("accuracy"), delegate
                {
                    evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluationStep);
            return (evaluationStep, _prediction);
        }

        private void UpdateTransferLearningModelOnDisk(DnnEstimator.Options options, int classCount)
        {
            var (sess, _, _, _) = BuildEvaluationSession(options, classCount);
            var graph = sess.graph;
            var outputGraphDef = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { _softMaxTensor.name.Split(':')[0], _prediction.name.Split(':')[0] });

            string frozenModelPath = _checkpointPath + ".pb";
            File.WriteAllBytes(_checkpointPath + ".pb", outputGraphDef.ToByteArray());
            _session = LoadTFSessionByModelFilePath(_env, frozenModelPath, false);
        }

        private void VariableSummaries(RefVariable var)
        {
            tf_with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                tf_with(tf.name_scope("stddev"), delegate
                {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }

        private (Operation, Tensor, Tensor, Tensor) AddObjectDetectionFinalRetrainOps(int classCount, string labelColumn,
            string scoreColumnName, float learningRate, Tensor bottleneckTensor, bool isTraining)
        {
            var anchor_per_scale = Configuration.YOLO_ANCHOR_PER_SCALE;
            OD_Utils utils = new OD_Utils();
            var classes = utils.read_class_names(Configuration.YOLO_CLASSES);
            var num_classes = classes.Length;
            var learn_rate_init = Configuration.TRAIN_LEARN_RATE_INIT;
            var learn_rate_end = Configuration.TRAIN_LEARN_RATE_END;
            var first_stage_epochs = Configuration.TRAIN_FISRT_STAGE_EPOCHS;
            var second_stage_epochs = Configuration.TRAIN_SECOND_STAGE_EPOCHS;
            var warmup_periods = Configuration.TRAIN_WARMUP_EPOCHS;
            var initial_weight = Configuration.TRAIN_INITIAL_WEIGHT;
            var moving_ave_decay = Configuration.YOLO_MOVING_AVE_DECAY;
            var max_bbox_per_scale = 150;
            var train_logdir = "./data/log/train";
            var trainset = Dataset('train');
            var testset = Dataset('test');
            var steps_per_period = len(self.trainset);

            Tensor input_data = null, label_sbbox = null, label_mbbox = null, label_lbbox = null, true_sbboxes = null, true_mbboxes = null, true_lbboxes = null, trainable = null;
            YoloV3 model;
            VariableV1[] net_var;
            Tensor giou_loss, conf_loss, prob_loss, loss;
            RefVariable global_step;

            tf_with(tf.name_scope("define_input"), scope =>
            {
                input_data = tf.placeholder(dtype: tf.float32, name: "input_data");
                label_sbbox = tf.placeholder(dtype: tf.float32, name: "label_sbbox");
                label_mbbox = tf.placeholder(dtype: tf.float32, name: "label_mbbox");
                label_lbbox = tf.placeholder(dtype: tf.float32, name: "label_lbbox"); ;
                true_sbboxes = tf.placeholder(dtype: tf.float32, name: "sbboxes");
                true_mbboxes = tf.placeholder(dtype: tf.float32, name: "mbboxes");
                true_lbboxes = tf.placeholder(dtype: tf.float32, name: "lbboxes");
                trainable = tf.placeholder(TF_DataType.TF_BOOL, name: "training");
            });

            tf_with(tf.name_scope("define_loss"), scope =>
            {
                model = new YoloV3(input_data, (bool)trainable);
                net_var = tf.global_variables();
                (giou_loss, conf_loss, prob_loss) = model.compute_loss(label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes);
                loss = giou_loss + conf_loss + prob_loss;
            });

            tf_with(tf.name_scope("learn_rate"), scope =>
            {
                global_step = tf.Variable(1.0, dtype: tf.float64, trainable: false, name: "global_step");
                var warmup_steps = tf.constant(warmup_periods * steps_per_period,
                                        dtype: tf.float64, name: "warmup_steps");
                var train_steps = tf.constant((first_stage_epochs + second_stage_epochs) * steps_per_period,
                                        dtype: tf.float64, name: "train_steps");
            });

            var (batch_size, bottleneck_tensor_size) = (bottleneckTensor.TensorShape.Dimensions[0], bottleneckTensor.TensorShape.Dimensions[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                _labelTensor = tf.placeholder(tf.int64, new TensorShape(batch_size), name: labelColumn);
            });

            string layerName = "final_retrain_ops";
            Tensor logits = null;
            tf_with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: 0.001f);
                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    VariableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                tf_with(tf.name_scope("biases"), delegate
                {
                    layerBiases = tf.Variable(tf.zeros(classCount), name: "final_biases");
                    VariableSummaries(layerBiases);
                });

                tf_with(tf.name_scope("Wx_plus_b"), delegate
                {
                    var matmul = tf.matmul(bottleneckTensor, layerWeights);
                    logits = matmul + layerBiases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            _softMaxTensor = tf.nn.softmax(logits, name: scoreColumnName);

            tf.summary.histogram("activations", _softMaxTensor);
            if (!isTraining)
                return (null, null, _labelTensor, _softMaxTensor);

            Tensor crossEntropyMean = null;
            tf_with(tf.name_scope("cross_entropy"), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: _labelTensor, logits: logits);
            });

            tf.summary.scalar("cross_entropy", crossEntropyMean);

            tf_with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.AdamOptimizer(learningRate);
                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _labelTensor, _softMaxTensor);
        }

        private (Operation, Tensor, Tensor, Tensor) AddFinalRetrainOps(int classCount, string labelColumn,
            string scoreColumnName, float learningRate, Tensor bottleneckTensor, bool isTraining)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensor.TensorShape.Dimensions[0], bottleneckTensor.TensorShape.Dimensions[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                _labelTensor = tf.placeholder(tf.int64, new TensorShape(batch_size), name: labelColumn);
            });

            string layerName = "final_retrain_ops";
            Tensor logits = null;
            tf_with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: 0.001f);
                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    VariableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                tf_with(tf.name_scope("biases"), delegate
                {
                    layerBiases = tf.Variable(tf.zeros(classCount), name: "final_biases");
                    VariableSummaries(layerBiases);
                });

                tf_with(tf.name_scope("Wx_plus_b"), delegate
                {
                    var matmul = tf.matmul(bottleneckTensor, layerWeights);
                    logits = matmul + layerBiases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            _softMaxTensor = tf.nn.softmax(logits, name: scoreColumnName);

            tf.summary.histogram("activations", _softMaxTensor);
            if (!isTraining)
                return (null, null, _labelTensor, _softMaxTensor);

            Tensor crossEntropyMean = null;
            tf_with(tf.name_scope("cross_entropy"), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: _labelTensor, logits: logits);
            });

            tf.summary.scalar("cross_entropy", crossEntropyMean);

            tf_with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(learningRate);
                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _labelTensor, _softMaxTensor);
        }

        private void AddTransferLearningLayer(string labelColumn,
            string scoreColumnName, float learningRate, int classCount)
        {
            _bottleneckTensor = Graph.OperationByName(_bottleneckOperationName);
            tf_with(Graph.as_default(), delegate
            {
                (_trainStep, _crossEntropy, _labelTensor, _softMaxTensor) =
                    AddFinalRetrainOps(classCount, labelColumn, scoreColumnName, learningRate, _bottleneckTensor, true);
            });
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, TensorShape tfShape, bool keyType = false)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            return new TensorValueGetter<T>(input, colIndex, tfShape, keyType);
        }

        private static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, TF_DataType tfType, bool isVector, int colIndex, TensorShape tfShape)
        {
            var type = DnnUtils.Tf2MlNetType(tfType);
            if (input.Schema[colIndex].Type is KeyDataViewType && type.RawType == typeof(Int64))
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, typeof(UInt32), input, isVector, colIndex, tfShape, true);

            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type.RawType, input, isVector, colIndex, tfShape, false);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(
            DataViewRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            TF_DataType[] tfInputTypes,
            TensorShape[] tfInputShapes)
        {
            var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                int colIndex = inputColIndices[i];
                srcTensorGetters[i] = CreateTensorValueGetter(input, tfInputTypes[i], isInputVector[i], colIndex, tfInputShapes[i]);
            }
            return srcTensorGetters;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs,
            out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput, out bool transferLearning,
            out string labelColumn, out string checkpointName, out Architecture arch,
            out string scoreColumnName, out string predictedColumnName, out float learningRate, out int classCount, out string predictionTensorName, out string softMaxTensorName)
        {
            isFrozen = ctx.Reader.ReadBoolByte();
            addBatchDimensionInput = ctx.Reader.ReadBoolByte();

            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            var numOutputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numOutputs > 0);
            outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();

            transferLearning = ctx.Reader.ReadBoolean();
            labelColumn = ctx.Reader.ReadString();
            checkpointName = ctx.Reader.ReadString();
            arch = (Architecture)ctx.Reader.ReadInt32();
            scoreColumnName = ctx.Reader.ReadString();
            predictedColumnName = ctx.Reader.ReadString();
            learningRate = ctx.Reader.ReadFloat();
            classCount = ctx.Reader.ReadInt32();
            predictionTensorName = ctx.Reader.ReadString();
            softMaxTensorName = ctx.Reader.ReadString();

        }

        internal DnnTransformer(IHostEnvironment env, Session session, string[] outputColumnNames,
            string[] inputColumnNames, string modelLocation, bool isTemporarySavedModel,
            bool addBatchDimensionInput, int batchSize, bool transferLearning, string labelColumnName, string checkpointName, Architecture arch,
            string scoreColumnName, string predictedLabelColumnName, float learningRate, DataViewSchema inputSchema, int? classCount = null, bool loadModel = false,
            string predictionTensorName = null, string softMaxTensorName = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            _env = env;
            _session = session;
            _modelLocation = modelLocation;
            _isTemporarySavedModel = isTemporarySavedModel;
            _addBatchDimensionInput = addBatchDimensionInput;
            _inputs = inputColumnNames;
            _outputs = outputColumnNames;
            _idvToTfMapping = new Dictionary<string, string>();
            _transferLearning = transferLearning;
            _labelColumnName = labelColumnName;
            _checkpointName = checkpointName;
            _arch = arch;
            _scoreColumnName = scoreColumnName;
            _predictedLabelColumnName = predictedLabelColumnName;
            _learningRate = learningRate;
            _softmaxTensorName = softMaxTensorName;
            _predictionTensorName = predictionTensorName;
            if (transferLearning)
            {
                if (classCount == null)
                {
                    var labelColumn = inputSchema.GetColumnOrNull(labelColumnName).Value;
                    var labelType = labelColumn.Type;
                    var labelCount = labelType.GetKeyCount();
                    if (labelCount <= 0)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", (string)labelColumn.Name, "Key", (string)labelType.ToString());

                    _classCount = labelCount == 1 ? 2 : (int)labelCount;
                }
                else
                    _classCount = classCount.Value;

                _checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), modelLocation + checkpointName);

                // Configure bottleneck tensor based on the model.
                if (arch == DnnEstimator.Architecture.ResnetV2101)
                    _bottleneckOperationName = "resnet_v2_101/SpatialSqueeze";
                else if (arch == DnnEstimator.Architecture.InceptionV3)
                    _bottleneckOperationName = "module_apply_default/hub_output/feature_vector/SpatialSqueeze";

                if (arch == DnnEstimator.Architecture.ResnetV2101)
                    _idvToTfMapping[_inputs[0]] = "input";
                else if (arch == DnnEstimator.Architecture.InceptionV3)
                    _idvToTfMapping[_inputs[0]] = "Placeholder";

                _outputs = new[] { scoreColumnName, predictedLabelColumnName };

                if (loadModel == false)
                {
                    // Add transfer learning layer.
                    AddTransferLearningLayer(labelColumnName, scoreColumnName, learningRate, _classCount);

                    // Initialize the variables.
                    new Runner(_session).AddOperation(tf.global_variables_initializer()).Run();

                    // Add evaluation layer.
                    (_evaluationStep, _) = AddEvaluationStep(_softMaxTensor, _labelTensor);
                    _softmaxTensorName = _softMaxTensor.name;
                    _predictionTensorName = _prediction.name;
                }

                _idvToTfMapping[scoreColumnName] = _softmaxTensorName;
                _idvToTfMapping[predictedLabelColumnName] = _predictionTensorName;

                (_tfOutputTypes, _outputTypes, _tfOutputOperations) = GetOutputInfo(Host, _session, new[] { _softmaxTensorName, _predictionTensorName });
                _transferLearning = true;
            }
            else
            {
                foreach (var x in _inputs)
                    _idvToTfMapping[x] = x;

                foreach (var x in _outputs)
                    _idvToTfMapping[x] = x;

                (_tfOutputTypes, _outputTypes, _tfOutputOperations) = GetOutputInfo(Host, _session, _outputs);

            }
            (_tfInputTypes, _tfInputShapes, _tfInputOperations) = GetInputInfo(Host, _session, _inputs.Select(x => _idvToTfMapping[x]).ToArray(), batchSize);

            _tfInputNodes = new TF_Output[_inputs.Length];
            _tfOutputNodes = new TF_Output[_outputs.Length];

            for (int index = 0; index < _tfInputOperations.Length; index += 1)
                _tfInputNodes[index] = new TF_Output(_tfInputOperations[index].Item1, _tfInputOperations[index].Item2);

            for (int index = 0; index < _tfOutputOperations.Length; index += 1)
                _tfOutputNodes[index] = new TF_Output(_tfOutputOperations[index].Item1, _tfOutputOperations[index].Item2);
        }

        private static (Operation, int) GetOperationFromName(string operation, Session session)
        {
            var p = operation.IndexOf(':');

            if (p != -1 && p != operation.Length - 1)
            {
                var op = operation.Substring(0, p);
                if (int.TryParse(operation.Substring(p + 1), out var idx))
                {

                    return (session.graph.OperationByName(op), idx);
                }
            }
            return (session.graph.OperationByName(operation), 0);
        }

        internal static (TF_DataType[] tfInputTypes, TensorShape[] tfInputShapes, (Operation, int)[]) GetInputInfo(IHost host, Session session, string[] inputs, int batchSize = 1)
        {
            var tfInputTypes = new TF_DataType[inputs.Length];
            var tfInputShapes = new TensorShape[inputs.Length];
            var tfInputOperations = new (Operation, int)[inputs.Length];

            int index = 0;
            foreach (var input in inputs)
            {
                host.CheckNonWhiteSpace(input, nameof(inputs));
                (Operation inputTensor, int inputTensorIndex) = GetOperationFromName(input, session);

                if (inputTensor == null)
                    throw host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");

                TF_DataType tfInputType = string.Compare(inputTensor.OpType, "PlaceHolder", true) == 0 ? inputTensor.OutputType(inputTensorIndex) : inputTensor.InputType(index);
                if (!DnnUtils.IsTypeSupported(tfInputType))
                    throw host.ExceptParam(nameof(session), $"Input type '{tfInputType}' of input column '{input}' is not supported in TensorFlow");

                tfInputTypes[index] = tfInputType;
                tfInputShapes[index] = ((Tensor)inputTensor).TensorShape;
                tfInputOperations[index] = (inputTensor, inputTensorIndex);
                index++;
            }

            return (tfInputTypes, tfInputShapes, tfInputOperations);
        }

        internal static TensorShape GetTensorShape(TF_Output output, Graph graph, Status status = null)
        {
            if (graph == IntPtr.Zero)
                new ObjectDisposedException(nameof(graph));

            var cstatus = status == null ? new Status() : status;
            var n = c_api.TF_GraphGetTensorNumDims(graph, output, cstatus);

            cstatus.Check();

            if (n == -1)
                return new TensorShape(new int[0]);

            var dims = new long[n];
            c_api.TF_GraphGetTensorShape(graph, output, dims, dims.Length, cstatus);
            cstatus.Check();
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        internal static (TF_DataType[] tfOutputTypes, DataViewType[] outputTypes, (Operation, int)[]) GetOutputInfo(IHost host, Session session, string[] outputs)
        {
            var tfOutputTypes = new TF_DataType[outputs.Length];
            var outputTypes = new DataViewType[outputs.Length];
            var newNames = new HashSet<string>();
            var tfOutputOperations = new (Operation, int)[outputs.Length];

            for (int i = 0; i < outputs.Length; i++)
            {
                host.CheckNonWhiteSpace(outputs[i], nameof(outputs));
                if (!newNames.Add(outputs[i]))
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' specified multiple times");

                (Tensor outputTensor, int outputIndex) = GetOperationFromName(outputs[i], session);
                if (outputTensor == null)
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' does not exist in the model");

                var tfOutputType = ((Operation)outputTensor).OutputType(outputIndex);
                var shape = GetTensorShape(new TF_Output((Operation)outputTensor, outputIndex), session.graph);

                // The transformer can only retreive the output as fixed length vector with shape of kind [-1, d1, d2, d3, ...]
                // i.e. the first dimension (if unknown) is assumed to be batch dimension.
                // If there are other dimension that are unknown the transformer will return a variable length vector.
                // This is the work around in absence of reshape transformer.
                int[] dims = shape.NDim > 0 ? shape.Dimensions.Skip(shape[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                for (int j = 0; j < dims.Length; j++)
                    dims[j] = dims[j] == -1 ? 0 : dims[j];
                if (dims == null || dims.Length == 0)
                {
                    dims = new[] { 1 };
                    outputTypes[i] = DnnUtils.Tf2MlNetType(tfOutputType);
                }
                else
                {
                    var type = DnnUtils.Tf2MlNetType(tfOutputType);
                    outputTypes[i] = new VectorDataViewType(type, dims);
                }

                tfOutputTypes[i] = tfOutputType;
                tfOutputOperations[i] = (outputTensor, outputIndex);
            }

            return (tfOutputTypes, outputTypes, tfOutputOperations);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            // stream: tensorFlow model.
            var isFrozen = _transferLearning || DnnUtils.IsSavedModel(_env, _modelLocation);
            ctx.Writer.WriteBoolByte(isFrozen);
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);

            Host.AssertNonEmpty(_inputs);
            ctx.Writer.Write(_inputs.Length);
            foreach (var colName in _inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(_outputs);
            ctx.Writer.Write(_outputs.Length);
            foreach (var colName in _outputs)
                ctx.SaveNonEmptyString(colName);

            ctx.Writer.Write(_transferLearning);
            ctx.Writer.Write(_labelColumnName);
            ctx.Writer.Write(_checkpointName);
            ctx.Writer.Write((int)_arch);
            ctx.Writer.Write(_scoreColumnName);
            ctx.Writer.Write(_predictedLabelColumnName);
            ctx.Writer.Write(_learningRate);
            ctx.Writer.Write(_classCount);
            ctx.Writer.Write(_predictionTensorName);
            ctx.Writer.Write(_softmaxTensorName);

            if (isFrozen || _transferLearning)
            {
                Status status = new Status();
                var buffer = _session.graph.ToGraphDef(status);
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.Data);
                });
            }
            else
            {
                ctx.SaveBinaryStream("TFSavedModel", w =>
                {
                    // only these files need to be saved.
                    string[] modelFilePaths =
                    {
                        Path.Combine(_modelLocation, DefaultModelFileNames.Graph),
                        Path.Combine(_modelLocation, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Data),
                        Path.Combine(_modelLocation, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Index),
                    };

                    w.Write(modelFilePaths.Length);

                    foreach (var fullPath in modelFilePaths)
                    {
                        var relativePath = fullPath.Substring(_modelLocation.Length + 1);
                        w.Write(relativePath);

                        using (var fs = new FileStream(fullPath, FileMode.Open))
                        {
                            long fileLength = fs.Length;
                            w.Write(fileLength);
                            long actualWritten = fs.CopyRange(w.BaseStream, fileLength);
                            Host.Assert(actualWritten == fileLength);
                        }
                    }
                });
            }
        }

        ~DnnTransformer()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            try
            {
                if (_session != null && _session != IntPtr.Zero)
                {
                    _session.close();
                }
            }
            finally
            {
                if (DnnUtils.IsSavedModel(_env, _modelLocation) && _isTemporarySavedModel)
                {
                    DnnUtils.DeleteFolderWithRetries(Host, _modelLocation);
                }
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly DnnTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TensorShape[] _fullySpecifiedShapes;
            private readonly ConcurrentBag<Runner> _runners;

            public Mapper(DnnTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent._inputs.Length];
                _isInputVector = new bool[_parent._inputs.Length];
                _fullySpecifiedShapes = new TensorShape[_parent._inputs.Length];
                for (int i = 0; i < _parent._inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent._inputs[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent._inputs[i]);

                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (type is VectorDataViewType vecType && vecType.Size == 0)
                        throw Host.Except("Variable length input columns not supported");

                    _isInputVector[i] = type is VectorDataViewType;
                    if (!_isInputVector[i])
                        throw Host.Except("Non-vector columns are not supported and should be loaded as vector columns of size 1");
                    vecType = (VectorDataViewType)type;
                    var expectedType = DnnUtils.Tf2MlNetType(_parent._tfInputTypes[i]);
                    if (type.GetItemType() != expectedType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent._inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent._tfInputShapes[i];
                    var shape = originalShape.Dimensions;

                    var colTypeDims = vecType.Dimensions.Select(dim => (int)dim).ToArray();
                    if (shape == null || (shape.Length == 0))
                        _fullySpecifiedShapes[i] = new TensorShape(colTypeDims);
                    else
                    {
                        // If the column is one dimension we make sure that the total size of the TF shape matches.
                        // Compute the total size of the known dimensions of the shape.
                        int valCount = 1;
                        int numOfUnkDim = 0;
                        foreach (var s in shape)
                        {
                            if (s > 0)
                                valCount *= s;
                            else
                                numOfUnkDim++;
                        }
                        // The column length should be divisible by this, so that the other dimensions can be integral.
                        int typeValueCount = type.GetValueCount();
                        if (typeValueCount % valCount != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent._inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. For example, if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = numOfUnkDim > 0 ? Math.Pow(typeValueCount / valCount, 1.0 / numOfUnkDim) : 0;
                        if (d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent._inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // Fill in the unknown dimensions.
                        var l = new int[originalShape.NDim];
                        for (int ishape = 0; ishape < originalShape.NDim; ishape++)
                            l[ishape] = originalShape[ishape] == -1 ? (int)d : originalShape[ishape];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }

                    if (_parent._addBatchDimensionInput)
                    {
                        var l = new int[_fullySpecifiedShapes[i].NDim + 1];
                        l[0] = 1;
                        for (int ishape = 1; ishape < l.Length; ishape++)
                            l[ishape] = _fullySpecifiedShapes[i][ishape - 1];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }
                }

                _runners = new ConcurrentBag<Runner>();
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private class OutputCache
            {
                public long Position;
                public Dictionary<string, Tensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, Tensor>();
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent._outputs.Where((x, i) => activeOutput(i)).ToArray();

                var type = DnnUtils.Tf2MlNetType(_parent._tfOutputTypes[iinfo]).RawType;
                Host.Assert(type == _parent._outputTypes[iinfo].GetItemType().RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent._tfInputTypes, _fullySpecifiedShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);

                if (_parent._outputTypes[iinfo].IsStandardScalar())
                {
                    ValueGetter<T> valuegetter = (ref T dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                        dst = tensor.Data<T>()[0];
                    };
                    return valuegetter;
                }
                else
                {
                    if (_parent._tfOutputTypes[iinfo] == TF_DataType.TF_STRING)
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.Dimensions.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                            DnnUtils.FetchStringData(tensor, editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                    else
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.Dimensions.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);

                            DnnUtils.FetchData<T>(tensor.Data<T>(), editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                }
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    Runner runner = new Runner(_parent._session);

                    // Feed the inputs.
                    for (int i = 0; i < _parent._inputs.Length; i++)
                        runner.AddInput(_parent._idvToTfMapping[_parent._inputs[i]], srcTensorGetters[i].GetTensor());

                    // Add outputs.
                    for (int i = 0; i < _parent._outputs.Length; i++)
                        runner.AddOutputs(_parent._idvToTfMapping[_parent._outputs[i]]);

                    // Execute the graph.
                    var tensors = runner.Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < activeOutputColNames.Length; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];

                    outputCache.Position = position;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent._outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent._outputs.Length];
                for (int i = 0; i < _parent._outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent._outputs[i], _parent._outputTypes[i], null);
                return info;
            }
        }

        private interface ITensorValueGetter
        {
            Tensor GetTensor();

            void BufferTrainingData();

            Tensor GetBufferedBatchTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly T[] _bufferedData;
            private readonly Int64[] _bufferedDataLong;
            private readonly TensorShape _tfShape;
            private int _position;
            private readonly bool _keyType;
            private long[] _dims;

            public TensorValueGetter(DataViewRow input, int colIndex, TensorShape tfShape, bool keyType = false)
            {
                _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
                _tfShape = tfShape;
                long size = 0;
                _position = 0;
                if (tfShape.Dimensions.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.Dimensions)
                        size *= dim;
                    _dims = _tfShape.Dimensions.Select(x => (long)x).ToArray();
                }
                if (keyType)
                    _bufferedDataLong = new long[size];
                else
                    _bufferedData = new T[size];
                _keyType = keyType;
            }

            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                if (_keyType)
                {
                    var tensor = new Tensor(new[] { Convert.ToInt64(scalar) - 1 });
                    tensor.SetShape(_tfShape);
                    return tensor;
                }
                else
                {
                    var tensor = new Tensor(new[] { scalar });
                    tensor.SetShape(_tfShape);
                    return tensor;
                }
            }

            public void BufferTrainingData()
            {
                if (_keyType)
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedDataLong[_position++] = Convert.ToInt64(scalar) - 1;
                }
                else
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedData[_position++] = scalar;
                }
            }

            public Tensor GetBufferedBatchTensor()
            {
                if (_keyType)
                {
                    var tensor = new Tensor(_bufferedDataLong, _dims, TF_DataType.TF_INT64);
                    _position = 0;
                    return tensor;
                }
                else
                {
                    var tensor = CastDataAndReturnAsTensor(_bufferedData);
                    _position = 0;
                    return tensor;
                }
            }

            private Tensor CastDataAndReturnAsTensor(T[] data)
            {
                if (typeof(T) == typeof(sbyte))
                    return new Tensor((sbyte[])(object)data, _dims, TF_DataType.TF_INT8);
                else if (typeof(T) == typeof(long))
                    return new Tensor((long[])(object)data, _dims, TF_DataType.TF_INT64);
                else if (typeof(T) == typeof(Int32))
                    return new Tensor((Int32[])(object)data, _dims, TF_DataType.TF_INT32);
                else if (typeof(T) == typeof(Int16))
                    return new Tensor((Int16[])(object)data, _dims, TF_DataType.TF_INT16);
                else if (typeof(T) == typeof(byte))
                    return new Tensor((byte[])(object)data, _dims, TF_DataType.TF_UINT8);
                else if (typeof(T) == typeof(ulong))
                    return new Tensor((ulong[])(object)data, _dims, TF_DataType.TF_UINT64);
                else if (typeof(T) == typeof(UInt32))
                    return new Tensor((UInt32[])(object)data, _dims, TF_DataType.TF_UINT32);
                else if (typeof(T) == typeof(UInt16))
                    return new Tensor((UInt16[])(object)data, _dims, TF_DataType.TF_UINT16);
                else if (typeof(T) == typeof(bool))
                    return new Tensor((bool[])(object)data, _dims, TF_DataType.TF_BOOL);
                else if (typeof(T) == typeof(float))
                    return new Tensor((float[])(object)data, _dims, TF_DataType.TF_FLOAT);
                else if (typeof(T) == typeof(float))
                    return new Tensor((double[])(object)data, _dims, TF_DataType.TF_DOUBLE);
                else if (typeof(T) == typeof(ReadOnlyMemory<char>))
                {
                    byte[][] bytes = new byte[_bufferedData.Length][];
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        bytes[i] = Encoding.UTF8.GetBytes(((ReadOnlyMemory<char>)(object)data[i]).ToArray());
                    }

                    return new Tensor(bytes, _tfShape.dims.Select(x => (long)x).ToArray());
                }

                return new Tensor(new NDArray(data, _tfShape));
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TensorShape _tfShape;
            private VBuffer<T> _vBuffer;
            private T[] _denseData;
            private T[] _bufferedData;
            private int _position;
            private long[] _dims;
            private readonly long _bufferedDataSize;

            public TensorValueGetterVec(DataViewRow input, int colIndex, TensorShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                _tfShape = tfShape;
                _vBuffer = default;
                _denseData = default;

                long size = 0;
                _position = 0;
                if (tfShape.Dimensions.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.Dimensions)
                        size *= dim;
                }
                _bufferedData = new T[size];
                _bufferedDataSize = size;
                if (_tfShape.Dimensions != null)
                    _dims = _tfShape.Dimensions.Select(x => (long)x).ToArray();
            }

            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is executed. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                _denseData = new T[_vBuffer.Length];
                _vBuffer.CopyTo(_denseData);
                return CastDataAndReturnAsTensor(_denseData);
            }

            private Tensor CastDataAndReturnAsTensor(T[] data)
            {
                if (typeof(T) == typeof(sbyte))
                    return new Tensor((sbyte[])(object)data, _dims, TF_DataType.TF_INT8);
                else if (typeof(T) == typeof(long))
                    return new Tensor((long[])(object)data, _dims, TF_DataType.TF_INT64);
                else if (typeof(T) == typeof(Int32))
                    return new Tensor((Int32[])(object)data, _dims, TF_DataType.TF_INT32);
                else if (typeof(T) == typeof(Int16))
                    return new Tensor((Int16[])(object)data, _dims, TF_DataType.TF_INT16);
                else if (typeof(T) == typeof(byte))
                    return new Tensor((byte[])(object)data, _dims, TF_DataType.TF_UINT8);
                else if (typeof(T) == typeof(ulong))
                    return new Tensor((ulong[])(object)data, _dims, TF_DataType.TF_UINT64);
                else if (typeof(T) == typeof(UInt32))
                    return new Tensor((UInt32[])(object)data, _dims, TF_DataType.TF_UINT32);
                else if (typeof(T) == typeof(UInt16))
                    return new Tensor((UInt16[])(object)data, _dims, TF_DataType.TF_UINT16);
                else if (typeof(T) == typeof(bool))
                    return new Tensor((bool[])(object)data, _dims, TF_DataType.TF_BOOL);
                else if (typeof(T) == typeof(float))
                    return new Tensor((float[])(object)data, _dims, TF_DataType.TF_FLOAT);
                else if (typeof(T) == typeof(double))
                    return new Tensor((double[])(object)data, _dims, TF_DataType.TF_DOUBLE);
                else if (typeof(T) == typeof(ReadOnlyMemory<char>))
                {
                    byte[][] bytes = new byte[_vBuffer.Length][];
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        bytes[i] = Encoding.UTF8.GetBytes(((ReadOnlyMemory<char>)(object)data[i]).ToArray());
                    }

                    return new Tensor(bytes, _tfShape.dims.Select(x => (long)x).ToArray());
                }

                return new Tensor(new NDArray(data, _tfShape));
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public Tensor GetBufferedBatchTensor()
            {
                _position = 0;
                var tensor = CastDataAndReturnAsTensor(_bufferedData);
                _bufferedData = new T[_bufferedDataSize];
                return tensor;
            }
        }
    }

    // OD Change - Convolution methods for YoloV3 Object Class
    internal class Common
    {
        public Tensor convolutional(Tensor input_data, TensorShape filters_shape, bool trainable, string name, bool activate = true, bool bn = true)
        {
            Tensor conv = tf.random_normal(null); // For return error

            tf_with(tf.variable_scope(name), delegate
            {
                var strides = new int[] { 1, 1, 1, 1 };
                var padding = "SAME";
                var weight = tf.get_variable(name: "weight", shape: filters_shape, dtype: tf.float32, initializer: tf.random_normal(filters_shape), true);
                conv = tf.nn.conv2d(input_data, weight, strides, padding);

                // Batch Normalization
                if (bn)
                {
                    conv = tf.layers.batch_normalization(conv, beta_initializer: tf.zeros_initializer,
                        gamma_initializer: tf.ones_initializer, moving_mean_initializer: tf.zeros_initializer, moving_variance_initializer: tf.ones_initializer, trainable: trainable);
                }
                else
                {
                    // Replace filters_shape[-1] with Slice implementation
                    var bias = tf.get_variable(name = "bias", shape: filters_shape[new Slice(filters_shape.dims.Length - 1, filters_shape.dims.Length)], trainable: true,
                                   dtype: tf.float32, initializer: tf.truncated_normal_initializer((float)0.0));
                    conv = tf.nn.bias_add(conv, bias);
                }

                // Activation
                if (activate) conv = tf_leakyrelu(conv, (float)0.1);
            });
            return conv;
        }

        public Tensor tf_leakyrelu(Tensor conv, float alpha)
        {
            return tf.nn.relu(conv) - alpha * tf.nn.relu(-1 * conv);
        }

        public Tensor residual_block(Tensor input_data, int input_channel, int filter_num1, int filter_num2, bool trainable, string name)
        {
            Tensor residual_output = tf.random_normal(null); // For return error

            var short_cut = input_data;
            tf_with(tf.variable_scope(name), delegate
            {
                input_data = convolutional(input_data, new TensorShape(1, 1, input_channel, filter_num1),
                                   trainable: trainable, name = "conv1");
                input_data = convolutional(input_data, new TensorShape(3, 3, filter_num1, filter_num2),
                                           trainable: trainable, name = "conv2");

                residual_output = input_data + short_cut;
            });
            return residual_output;
        }

        public Tensor route(string name, Tensor previous_output, Tensor current_output)
        {
            Tensor output = tf.random_normal(null); // For return error

            tf_with(tf.variable_scope(name), delegate
            {
                output = tf.concat(new List<Tensor> { current_output, previous_output }, axis: -1);
            });
            return output;
        }

        public Tensor upsample(Tensor input_data, string name, string method = "deconv")
        {
            Tensor output = tf.random_normal(null); // For return error

            if (method.Equals("resize"))
            {
                tf_with(tf.variable_scope(name), delegate
                {
                    var input_shape = input_data.TensorShape;
                    output = tf.image.resize_bilinear(input_data, new Tensor(new int[] { input_shape[1] * 2, input_shape[2] * 2 }));
                });
            } else
            {
                // replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
                var num_filter = input_data.TensorShape.dims.Length;
                output = tf.layers.conv2d(input_data, num_filter, kernel_size: new int[] { 2 }, padding: "same",
                    strides: new int[] { 2, 2 }, kernel_initializer: tf.zeros_initializer);
            }
            return output;
        }
    }

    // OD Change - Darknet YoloV3 Implementation
    internal class Darknet53
    {
        public (Tensor, Tensor, Tensor) darknet53(Tensor input_data, bool trainable)
        {
            tf_with(tf.variable_scope("darknet"), delegate
            {
                Common common = new Common();
                input_data = common.convolutional(input_data, filters_shape: (3, 3, 3, 32), trainable: trainable, name: "conv0");
                input_data = common.convolutional(input_data, filters_shape: (3, 3, 32, 64),
                    trainable = trainable, name: "conv1");

                for (int i = 0; i < 1; i++)
                {
                    input_data = common.residual_block(input_data, 64, 32, 64, trainable: trainable, name: textmod("residual%d", i));
                }

                input_data = common.convolutional(input_data, filters_shape: (3, 3, 64, 128), trainable: trainable, name: "conv4");

                for (int i = 0; i < 2; i++)
                {
                    input_data = common.residual_block(input_data, 128, 64, 128, trainable: trainable, name: textmod("residual%d", i + 1));
                }

                input_data = common.convolutional(input_data, filters_shape: (3, 3, 128, 256), trainable: trainable, name: "conv4");

                for (int i = 0; i < 8; i++)
                {
                    input_data = common.residual_block(input_data, 256, 128, 256, trainable: trainable, name: textmod("residual%d", i + 3));
                }
                var route_1 = input_data;
                input_data = common.convolutional(input_data, filters_shape: (3, 3, 256, 512), trainable: trainable, name: "conv4");

                for (int i = 0; i < 8; i++)
                {
                    input_data = common.residual_block(input_data, 512, 256, 512, trainable: trainable, name: textmod("residual%d", i + 11));
                }
                var route_2 = input_data;
                input_data = common.convolutional(input_data, filters_shape: (3, 3, 512, 1024), trainable: trainable, name: "conv4");

                for (int i = 0; i < 4; i++)
                {
                    input_data = common.residual_block(input_data, 1024, 512, 1024, trainable: trainable, name: textmod("residual%d", i + 19));
                }

                return (route_1, route_2, input_data);
            });
            return (null, null, null); // For return error
        }

        public string textmod(string str, int i)
        {
            return str.Substring(0, str.IndexOf("%")) + i;
        }
    }

    // OD Change: Model Configuration
    internal class Configuration
    {
        // YOLO Constants

        public const string YOLO_CLASSES                = "./data/classes/coco.names";
        public const string YOLO_ANCHORS                = "./data/anchors/basline_anchors.txt";
        public const double YOLO_MOVING_AVE_DECAY       = 0.9995;
        public static int[] YOLO_STRIDES                = new int[] {8, 16, 32};
        public const int    YOLO_ANCHOR_PER_SCALE       = 3;
        public const double YOLO_IOU_LOSS_THRESH        = 0.5;
        public const string YOLO_UPSAMPLE_METHOD        = "resize";
        public const string YOLO_ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt";
        public const string YOLO_DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt";

        // TRAIN Constants

        public const string TRAIN_ANNOT_PATH                  = "./data/dataset/voc_train.txt";
        public const double TRAIN_BATCH_SIZE                  = 6;
        public static int[]        TRAIN_INPUT_SIZE                  = new int[] {320, 352, 384, 416, 448, 480, 512, 544, 576, 608};
        public const bool   TRAIN_DATA_AUG                    = true;
        public const double TRAIN_LEARN_RATE_INIT             = 1e-4;
        public const double TRAIN_LEARN_RATE_END              = 1e-6;
        public const double TRAIN_WARMUP_EPOCHS               = 2;
        public const double TRAIN_FISRT_STAGE_EPOCHS          = 20;
        public const double TRAIN_SECOND_STAGE_EPOCHS         = 30;
        public const string TRAIN_INITIAL_WEIGHT              = "./checkpoint/yolov3_coco_demo.ckpt";

        // TEST Constants

        public const string TEST_ANNOT_PATH                   = "./data/dataset/voc_test.txt";
        public const int    TEST_BATCH_SIZE                   = 2;
        public const int    TEST_INPUT_SIZE                   = 544;
        public const bool   TEST_DATA_AUG                     = false;
        public const bool   TEST_WRITE_IMAGE                  = true;
        public const string TEST_WRITE_IMAGE_PATH             = "./data/detection/";
        public const bool   TEST_WRITE_IMAGE_SHOW_LABEL       = true;
        public const string TEST_WEIGHT_FILE                  = "./checkpoint/yolov3_test_loss=9.2099.ckpt-5";
        public const bool   TEST_SHOW_LABEL                   = true;
        public const double TEST_SCORE_THRESHOLD              = 0.3;
        public const double TEST_IOU_THRESHOLD                = 0.45;

    }

    // OD Change: Model Constructor
    internal class YoloV3
    {
        public bool trainable;
        static OD_Utils utils = new OD_Utils();
        public static string[] classes = utils.read_class_names(Configuration.YOLO_CLASSES);
        public int num_classes = classes.Length;
        public NDArray strides = np.array(Configuration.YOLO_STRIDES);
        public int anchor_per_scale = Configuration.YOLO_ANCHOR_PER_SCALE;
        public double iou_loss_thresh = Configuration.YOLO_IOU_LOSS_THRESH;
        public string upsample_method = Configuration.YOLO_UPSAMPLE_METHOD;
        public Tensor conv_lbbox, conv_mbbox, conv_sbbox;
        public NDArray anchors = utils.get_anchors(Configuration.YOLO_ANCHORS);
        public Tensor pred_sbbox, pred_mbbox, pred_lbbox;
        public YoloV3(Tensor input_data, bool trainable)
        {
            this.trainable = trainable;
            try
            {
                (conv_lbbox, conv_mbbox, conv_sbbox) = build_network(input_data);
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("Could not build YOLOv3 network.");
            }

            tf_with(tf.variable_scope("pred_sbbox"), delegate
            {
                pred_sbbox = decode(conv_sbbox, anchors[0], strides[0]);
            });

            tf_with(tf.variable_scope("pred_mbbox"), delegate
            {
                pred_mbbox = decode(conv_mbbox, anchors[1], strides[1]);
            });

            tf_with(tf.variable_scope("pred_lbbox"), delegate
            {
                pred_lbbox = decode(conv_lbbox, anchors[2], strides[2]);
            });
        }

        public (Tensor, Tensor, Tensor) build_network(Tensor input_data)
        {
            Darknet53 backbone = new Darknet53();
            Tensor route_1, route_2;
            (route_1, route_2, input_data) = backbone.darknet53(input_data, trainable);

            Common common = new Common();
            input_data = common.convolutional(input_data, (1, 1, 1024, 512), trainable, "conv52");
            input_data = common.convolutional(input_data, (3, 3, 512, 1024), trainable, "conv53");
            input_data = common.convolutional(input_data, (1, 1, 1024, 512), trainable, "conv54");
            input_data = common.convolutional(input_data, (3, 3, 512, 1024), trainable, "conv55");
            input_data = common.convolutional(input_data, (1, 1, 1024, 512), trainable, "conv56");

            Tensor conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), trainable, name: "conv_lobj_branch");
            Tensor conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (num_classes + 5)),
                trainable: trainable, name: "conv_lbbox", bn: false, activate: false);

            input_data = common.convolutional(input_data, (1, 1, 512, 256), trainable, "conv57");
            input_data = common.upsample(input_data, name: "upsample0", method: upsample_method);

            tf_with(tf.variable_scope("route_1"), delegate
            {
                List<Tensor> tensorList = new List<Tensor>();
                tensorList.Insert(0, input_data);
                tensorList.Insert(1, route_2);
                input_data = tf.concat(tensorList, axis: -1);
            });

            input_data = common.convolutional(input_data, (1, 1, 768, 256), trainable, "conv58");
            input_data = common.convolutional(input_data, (3, 3, 256, 512), trainable, "conv59");
            input_data = common.convolutional(input_data, (1, 1, 512, 256), trainable, "conv60");
            input_data = common.convolutional(input_data, (3, 3, 256, 512), trainable, "conv61");
            input_data = common.convolutional(input_data, (1, 1, 512, 256), trainable, "conv62");

            Tensor conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), trainable, name: "conv_mobj_branch");
            Tensor conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (num_classes + 5)),
                                          trainable: trainable, name: "conv_mbbox", activate: false, bn: false);

            input_data = common.convolutional(input_data, (1, 1, 256, 128), trainable, "conv63");
            input_data = common.upsample(input_data, name: "upsample1", method: upsample_method);

            tf_with(tf.variable_scope("route_2"), delegate
            {
                List<Tensor> tensorList = new List<Tensor>();
                tensorList.Insert(0, input_data);
                tensorList.Insert(1, route_1);
                input_data = tf.concat(tensorList, axis: -1);
            });

            input_data = common.convolutional(input_data, (1, 1, 384, 128), trainable, "conv64");
            input_data = common.convolutional(input_data, (3, 3, 128, 256), trainable, "conv65");
            input_data = common.convolutional(input_data, (1, 1, 256, 128), trainable, "conv66");
            input_data = common.convolutional(input_data, (3, 3, 128, 256), trainable, "conv67");
            input_data = common.convolutional(input_data, (1, 1, 256, 128), trainable, "conv68");

            Tensor conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), trainable, name: "conv_sobj_branch");
            Tensor conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (num_classes + 5)),
                                          trainable = trainable, name: "conv_sbbox", activate: false, bn: false);

            return (conv_lbbox, conv_mbbox, conv_sbbox);
        }

        public Tensor decode(Tensor conv_output, NDArray anchors, NDArray stride)
        {
            var conv_shape = conv_output.TensorShape;
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            var anchor_per_scale = anchors.len;

            conv_output = tf.reshape(conv_output, new int[] {batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes});

            //var conv_raw_dxdy = conv_output[:, :, :, :, 0:2];
            //var conv_raw_dwdh = conv_output[:, :, :, :, 2:4];
            //var conv_raw_conf = conv_output[:, :, :, :, 4:5];
            //var conv_raw_prob = conv_output[:, :, :, :, 5: ];
            // should be tf.slice, strided_slice with step 1 should be equivilant
            //var ex_raw_dxdy = tf.slice(conv_output, [0, 0, 0, 0, 0], [batch_size, output_size, output_size, anchor_per_scale, 2]);
            Tensor s1 = new Tensor(new[] { 0, 0, 0, 0, 0 });
            Tensor e1 = new Tensor(new[] { batch_size, output_size, output_size, anchor_per_scale, 2 });
            Tensor s2 = new Tensor(new[] { 0, 0, 0, 0, 2 });
            Tensor e2 = new Tensor(new[] { batch_size, output_size, output_size, anchor_per_scale, 3 });
            Tensor s3 = new Tensor(new[] { 0, 0, 0, 0, 4 });
            Tensor e3 = new Tensor(new[] { batch_size, output_size, output_size, anchor_per_scale, 5 });
            Tensor s4 = new Tensor(new[] { 0, 0, 0, 0, 5 });
            Tensor e4 = new Tensor(new[] { batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes });
            Tensor s = new Tensor(new[] { 1, 1, 1, 1, 1 });
            var conv_raw_dxdy = tf.strided_slice(conv_output, s1, e1, s);
            var conv_raw_dwdh = tf.strided_slice(conv_output, s2, e2, s);
            var conv_raw_conf = tf.strided_slice(conv_output, s3, e3, s);
            var conv_raw_prob = tf.strided_slice(conv_output, s4, e4, s);

            Int32[] range = new Int32[output_size];
            for (int i = 0; i < output_size; i++) { range[i] = i; }
            Tensor range_tensor = new Tensor(range, dType: tf.int32);

            var y = tf.tile(tf.expand_dims(range_tensor, -1), new Tensor(new[] { 1, output_size }));
            var x = tf.tile(tf.expand_dims(range_tensor, 1), new Tensor(new[] { output_size, 1 }));

            var xy_grid = tf.concat(new[] { tf.expand_dims(x, -1), tf.expand_dims(y, -1) }, axis: -1);
            xy_grid = tf.expand_dims(xy_grid, 0); xy_grid = tf.expand_dims(xy_grid, -2);
            xy_grid = tf.tile(xy_grid, new Tensor(new[] { batch_size, 1, 1, anchor_per_scale, 1 }));
            xy_grid = tf.cast(xy_grid, tf.float32);

            var stride_tensor = tf.convert_to_tensor(stride);
            var pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride_tensor;
            var pred_wh = (tf.exp(conv_raw_dwdh) * tf.convert_to_tensor(anchors)) * stride_tensor;
            var pred_xywh = tf.concat(new[] { pred_xy, pred_wh }, axis: -1);

            var pred_conf = tf.sigmoid(conv_raw_conf);
            var pred_prob = tf.sigmoid(conv_raw_prob);

            return tf.concat(new[] { pred_xywh, pred_conf, pred_prob }, axis: -1);
        }

        // Slices tensor with the specified start and end for each dimension
        private Tensor slice(Tensor input, int[] start, int[] end)
        {
            Tensor s = new Tensor(start);
            Tensor e = new Tensor(end);
            int[] stride = new int[input.TensorShape.dims.Length];
            for (int i = 0; i < stride.Length; i++) { stride[i] = 1;  }
            Tensor stride_tensor = new Tensor(stride);
            return tf.strided_slice(input, s, e, stride_tensor);
        }

        // Slices tensor's last dimension at the specified start and end
        private Tensor slice(Tensor input, int start, int end)
        {
            int[] s = new int[input.TensorShape.dims.Length];
            int[] e = new int[input.TensorShape.dims.Length];
            for (int i = 0; i < s.Length-1; i++)
            {
                s[i] = 0;
                e[i] = input.TensorShape.dims[i];
            }
            s[s.Length - 1] = start;
            e[e.Length - 1] = end > 0 ? end : input.TensorShape.dims[e.Length - 1];
            return slice(input, s, e);
        }

        public Tensor focal(Tensor target, Tensor actual, int alpha = 1, double gamma = 2)
        {
            return new Tensor(alpha * tf.pow(target - actual, gamma));
        }

        public Tensor bbox_giou(Tensor boxes1, Tensor boxes2)
        {
            boxes1 = tf.concat(new[] {slice(boxes1, 0, 2) - slice(boxes1, 2, -1) * 0.5,
                            slice(boxes1, 0, 2) - slice(boxes1, 2, -1) * 0.5 }, axis: -1);
            boxes2 = tf.concat(new[] {slice(boxes2, 0, 2) - slice(boxes2, 2, -1) * 0.5,
                                slice(boxes2, 0, 2) + slice(boxes2, 2, -1) * 0.5}, axis: -1);

            boxes1 = tf.concat(new[] {tf.minimum(slice(boxes1, 0, 2), slice(boxes1, 2, -1)),
                                tf.maximum(slice(boxes1, 0, 2), slice(boxes1, 2, -1)) }, axis: -1);
            boxes2 = tf.concat(new[] {tf.minimum(slice(boxes2, 0, 2), slice(boxes2, 2, -1)),
                                tf.maximum(slice(boxes2, 0, 2), slice(boxes2, 2, -1)) }, axis: -1);

            //boxes1[..., 2] boxes1[..., 0] boxes1[..., 3] boxes1[..., 1] boxes2[..., 2] boxes2[..., 0] boxes2[..., 3] boxes2[..., 1]
            var boxes1_area = (slice(boxes1, 2, 3) - slice(boxes1, 0, 1)) * (slice(boxes1, 3, 4) - slice(boxes1, 1, 2));
            var boxes2_area = (slice(boxes2, 2, 3) - slice(boxes2, 0, 1)) * (slice(boxes2, 3, 4) - slice(boxes2, 1, 2));

            var left_up = tf.maximum(slice(boxes1, 0, 2), slice(boxes2, 0, 2));
            var right_down = tf.minimum(slice(boxes1, 2, -1), slice(boxes2, 2, -1));

            var inter_section = tf.maximum(right_down - left_up, 0.0);
            var inter_area = slice(inter_section, 0, 1) * slice(inter_section, 1, 2);
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = inter_area / union_area;

            var enclose_left_up = tf.minimum(slice(boxes1, 0, 2), slice(boxes2, 0, 2));
            var enclose_right_down = tf.maximum(slice(boxes1, 2, -1), slice(boxes2, 2, -1));
            var enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0);
            var enclose_area = slice(enclose, 0, 1) * slice(enclose, 1, 2);
            var giou = iou - 1.0 * (enclose_area - union_area) / enclose_area;

            return giou;
        }

        public Tensor bbox_iou(Tensor boxes1, Tensor boxes2)
        {

            var boxes1_area = slice(boxes1, 2, 3) * slice(boxes1, 3, 4);
            var boxes2_area = slice(boxes2, 2, 3) * slice(boxes2, 3, 4);

            boxes1 = tf.concat(new[] {slice(boxes1, 0, 2) - slice(boxes1, 2, -1) * 0.5,
                             slice(boxes1, 0, 2) + slice(boxes1, 2, -1) * 0.5 }, axis: -1);
            boxes2 = tf.concat(new[] { slice(boxes2, 0, 2) - slice(boxes2, 2, -1) * 0.5,
                             slice(boxes2, 0, 2) + slice(boxes2, 2, -1) * 0.5}, axis: -1);

            var left_up = tf.maximum(slice(boxes1, 0, 2), slice(boxes2, 0, 2));
            var right_down = tf.minimum(slice(boxes1, 2, -1), slice(boxes2, 2, -1));

            var inter_section = tf.maximum(right_down - left_up, 0.0);
            var inter_area = slice(inter_section, 0, 1) * slice(inter_section, 1, 2);
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = 1.0 * inter_area / union_area;

            return iou;
        }

        public (Tensor, Tensor, Tensor) loss_layer(Tensor conv, Tensor pred, Tensor label, Tensor bboxes, NDArray anchors, NDArray stride)
        {
            var conv_shape = conv.TensorShape;
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            var input_size = stride * output_size; //NDArray
            conv = tf.reshape(conv, new TensorShape(batch_size, output_size, output_size,
                                 anchor_per_scale, 5 + num_classes));
            var conv_raw_conf = slice(conv, 4, 5);//conv[:, :, :, :, 4:5];
            var conv_raw_prob = slice(conv, 5, -1);//conv[:, :, :, :, 5:];

            var pred_xywh = slice(pred, 0, 4);//pred[:, :, :, :, 0:4];
            var pred_conf = slice(pred, 4, 5);//pred[:, :, :, :, 4:5];

            var label_xywh = slice(label, 0, 4);//label[:, :, :, :, 0:4];
            var respond_bbox = slice(label, 4, 5);//label[:, :, :, :, 4:5];
            var label_prob = slice(label, 5, -1);//label[:, :, :, :, 5:];

            var giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis: -1);
            var input_size_tensor = tf.cast(tf.convert_to_tensor(input_size), tf.float32);

            var bbox_loss_scale = 2.0 - 1.0 * slice(label_xywh, 2, 3) * slice(label_xywh, 3, 4) / (tf.pow(input_size_tensor, 2));
            var giou_loss = respond_bbox * bbox_loss_scale * (1 - giou);

            // Replacing np.newaxis with tf.expand_dims
            bboxes = tf.expand_dims(pred_xywh, -3);
            bboxes = tf.expand_dims(pred_xywh, -4);
            bboxes = tf.expand_dims(pred_xywh, -5);
            var iou = bbox_iou(tf.expand_dims(pred_xywh, -2), bboxes);
            var max_iou = tf.expand_dims(tf.reduce_max(new Tensor(iou), axis: new int[] { -1 }), axis: -1);

            var respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32);

            var conf_focal = focal(respond_bbox, pred_conf);

            var conf_loss = conf_focal * (
                respond_bbox * tf_sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf)
                +
                respond_bgd * tf_sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf)
            );

            var prob_loss = respond_bbox * tf_sigmoid_cross_entropy_with_logits(labels: label_prob, logits: conv_raw_prob);

            for (int i = 1; i < 5; i++)
            {
                giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis: i));
                conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis: i));
                prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis: i));
            }

            return (giou_loss, conf_loss, prob_loss);
        }

        public Tensor tf_sigmoid_cross_entropy_with_logits(Tensor labels, Tensor logits)
        {
            float x = (float)0;
            return tf.max(logits.ToTFDataType(x.GetType()), 0) - logits * labels + tf.log(1 + tf.exp(-tf.abs(logits)));
        }

        public (Tensor, Tensor, Tensor) compute_loss(Tensor label_sbbox, Tensor label_mbbox, Tensor label_lbbox, Tensor true_sbbox, Tensor true_mbbox, Tensor true_lbbox)
        {
            (Tensor, Tensor, Tensor) loss_sbbox = (null, null, null);
            (Tensor, Tensor, Tensor) loss_mbbox = (null, null, null);
            (Tensor, Tensor, Tensor) loss_lbbox = (null, null, null);
            Tensor giou_loss = null, conf_loss = null, prob_loss = null;
            tf_with(tf.variable_scope("smaller_box_loss"), delegate
            {
                loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                         anchors: anchors[0], stride: strides[0]);
            });

            tf_with(tf.variable_scope("medium_box_loss"), delegate
            {
                loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                         anchors: anchors[1], stride: strides[1]);
            });

            tf_with(tf.variable_scope("bigger_box_loss"), delegate
            {
                loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                         anchors: anchors[2], stride: strides[2]);
            });

            tf_with(tf.variable_scope("giou_loss"), delegate
            {
                giou_loss = loss_sbbox.Item1 + loss_mbbox.Item1 + loss_lbbox.Item1;
            });

            tf_with(tf.variable_scope("conf_loss"), delegate
            {
                conf_loss = loss_sbbox.Item2 + loss_mbbox.Item2 + loss_lbbox.Item2;
            });

            tf_with(tf.variable_scope("prob_loss"), delegate
            {
                prob_loss = loss_sbbox.Item3 + loss_mbbox.Item3 + loss_lbbox.Item3;
            });

            return (giou_loss, conf_loss, prob_loss);
        }
    }

    // OD Change: YOLOv3 Util Methods
    internal class OD_Utils
    {
        public string[] read_class_names(string class_file_name)
        {
            string[] names = new string[] {};
            int i = 0;
            foreach (string line in File.ReadLines(class_file_name, Encoding.UTF8))
            {
                names[i] = line.Substring(0, line.Length - 2);
                i++;
            }
            return names;
        }

        public NDArray get_anchors(string anchors_path)
        {
            string anchors = File.ReadAllText(anchors_path, Encoding.UTF8);
            return np.array(anchors.Split(','), dtype: np.float32).reshape(new int[] { 3, 3, 2 });
        }
    }

    // OD Change: Dataset processing
    internal class Dataset
    {
        string annot_path;
        int[] input_sizes;
        double batch_size;
        bool data_aug;
        int[] train_input_sizes;
        NDArray strides;
        string[] classes;
        int num_classes;
        NDArray anchors;
        int anchor_per_scale;
        int max_bbox_per_scale;
        int num_samples;
        int num_batchs;
        int batch_count;

        public Dataset(string dataset_type)
        {
            if (dataset_type == "train") annot_path = Configuration.TRAIN_ANNOT_PATH;
            else annot_path = Configuration.TEST_ANNOT_PATH;

            if (dataset_type == "train") input_sizes = Configuration.TRAIN_INPUT_SIZE;
            else input_sizes = Configuration.TEST_INPUT_SIZE;

            if (dataset_type == "train")  batch_size = Configuration.TRAIN_BATCH_SIZE;
            else batch_size = Configuration.TEST_BATCH_SIZE;

            if (dataset_type == "train") data_aug = Configuration.TRAIN_DATA_AUG;
            else data_aug = Configuration.TEST_DATA_AUG;

            train_input_sizes = Configuration.TRAIN_INPUT_SIZE;
            strides = np.array(Configuration.YOLO_STRIDES);

            OD_Utils utils = new OD_Utils();
            classes = utils.read_class_names(Configuration.YOLO_CLASSES);
            num_classes = classes.Length;
            anchors = np.array(utils.get_anchors(Configuration.YOLO_ANCHORS));
            anchor_per_scale = Configuration.YOLO_ANCHOR_PER_SCALE;
            max_bbox_per_scale = 150;

            annotations = load_annotations(dataset_type);
            num_samples = annotations.Length;
            num_batchs = int(np.ceil(num_samples / batch_size));
            batch_count = 0;
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="DnnTransformer"]/*' />
    public sealed class DnnEstimator : IEstimator<DnnTransformer>
    {
        /// <summary>
        /// Image classification model.
        /// </summary>
        public enum Architecture
        {
            ResnetV2101,
            InceptionV3,
            YoloV3 // OD Change
        };

        /// <summary>
        /// Backend DNN training framework.
        /// </summary>
        public enum DnnFramework
        {
            Tensorflow
        };

        /// <summary>
        /// The options for the <see cref="DnnTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// Location of the TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "TensorFlow model used by the transform. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", SortOrder = 0)]
            public string ModelLocation;

            /// <summary>
            /// The names of the model inputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            /// <summary>
            /// The names of the requested model outputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;

            /// <summary>
            /// The name of the label column in <see cref="IDataView"/> that will be mapped to label node in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training labels.", ShortName = "label", SortOrder = 4)]
            public string LabelColumn;

            /// <summary>
            /// The name of the label in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "TensorFlow label node.", ShortName = "TFLabel", SortOrder = 5)]
            public string TensorFlowLabel;

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for optimizing parameters in the graph.
            /// Usually it is the name specified in the minimize method of optimizer in python
            /// e.g. optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name = "SGDOptimizer").
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the optimization operation in the TensorFlow graph.", ShortName = "OptimizationOp", SortOrder = 6)]
            public string OptimizationOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute training loss (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute training loss (Optional)", ShortName = "LossOp", SortOrder = 7)]
            public string LossOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute performance metric during training (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute performance metric during training (Optional)", ShortName = "MetricOp", SortOrder = 8)]
            public string MetricOperation;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 64;

            /// <summary>
            /// Number of training iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations.", SortOrder = 10)]
            public int Epoch = 5;

            /// <summary>
            /// The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).", SortOrder = 11)]
            public string LearningRateOperation;

            /// <summary>
            /// Learning rate to use during optimization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Name of the input in TensorFlow graph that specifiy the location for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/Const'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 13)]
            public string SaveLocationOperation = "save/Const";

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/control_dependency'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 14)]
            public string SaveOperation = "save/control_dependency";

            /// <summary>
            /// Needed for command line to specify if retraining is requested.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Retrain TensorFlow model.", SortOrder = 15)]
            public bool ReTrain = false;

            /// <summary>
            /// Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
            /// </summary>
            /// <remarks>
            /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.
            /// In this case, there is no way to induce shape from the model's inputs or input data.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].", SortOrder = 16)]
            public bool AddBatchDimensionInputs = false;

            /// <summary>
            /// Indicates if transfer learning is requested.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Transfer learning on a model.", SortOrder = 15)]
            public bool TransferLearning = false;

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.ResnetV2101;

            /// <summary>
            /// Name of the tensor that will contain the output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Softmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string ScoreColumnName = "Scores";

            /// <summary>
            /// Name of the tensor that will contain the predicted label from output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Argmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string PredictedLabelColumnName = "PredictedLabel";

            /// <summary>
            /// Checkpoint folder to store graph files in the event of transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Checkpoint folder to store graph files in the event of transfer learning.", SortOrder = 15)]
            public string CheckpointName = "_retrain_checkpoint";

            /// <summary>
            /// Use train set to measure model accuracy between each epoch.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use train set to measure model accuracy between each epoch.", SortOrder = 15)]
            public bool MeasureTrainAccuracy = false;
        }

        private readonly IHost _host;
        private readonly Options _options;
        private readonly DnnModel _tensorFlowModel;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private DnnTransformer _transformer;

        internal DnnEstimator(IHostEnvironment env, Options options, DnnModel tensorFlowModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnEstimator));
            _options = options;
            _tensorFlowModel = tensorFlowModel;

            if (options.TransferLearning)
                _tfInputTypes = new[] { TF_DataType.TF_FLOAT };
            else
            {
                var inputTuple = DnnTransformer.GetInputInfo(_host, tensorFlowModel.Session, options.InputColumns);
                _tfInputTypes = inputTuple.tfInputTypes;
            }
            if (options.TransferLearning)
                _outputTypes = new[] { new VectorDataViewType(NumberDataViewType.Single), new VectorDataViewType(NumberDataViewType.Single, 1) };
            else
                _outputTypes = DnnTransformer.GetOutputInfo(_host, tensorFlowModel.Session, options.OutputColumns).outputTypes;
        }

        private static Options CreateArguments(DnnModel tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
            options.ReTrain = false;
            options.AddBatchDimensionInputs = addBatchDimensionInput;
            return options;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            for (var i = 0; i < _options.InputColumns.Length; i++)
            {
                var input = _options.InputColumns[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());
                var expectedType = DnnUtils.Tf2MlNetType(_tfInputTypes[i]);
                if (col.ItemType != expectedType)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }
            for (var i = 0; i < _options.OutputColumns.Length; i++)
            {
                resultDic[_options.OutputColumns[i]] = new SchemaShape.Column(_options.OutputColumns[i],
                    _outputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, _outputTypes[i].GetItemType(), false);
            }
            return new SchemaShape(resultDic.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="DnnTransformer"/>.
        /// </summary>
        public DnnTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
                _transformer =  new DnnTransformer(_host, _options, _tensorFlowModel, input);

            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}
