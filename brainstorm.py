import re
from pathlib import Path
import xmltodict

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Side, Alignment
from openpyxl.worksheet.merge import MergeCells

x = """<?xml version="1.0" encoding="utf-8"?>
<EIX-LIST xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <ITEMS>
                               <ELEMENT xsi:type="string">&lt;padding&gt;</ELEMENT>
                               <ELEMENT xsi:type="string">&lt;padding&gt;</ELEMENT>
                               <ELEMENT format-rev="1" id="ed1ed4c6-ffbf-4d9f-9263-e94301c0301c" xsi:type="noEventTestStepMappingContainer">
                                               <TESTSTEP format-rev="3" id="1fe4cfad-088a-47b0-9ef3-3d1dbb8438e5" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Precondition</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP format-rev="3" id="67f401e4-910b-46e6-ab03-56e068e5a6f9" xsi:type="tsPackage">
                                                                               <PACKAGE-REFERENCE xsi:type="expressionref">
                                                                                              <PATH-EXPRESSION format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="string">C:/HIL_AE/Packages/Packages/FuSi/EATR/../../lib/HiL Init.pkg</VALUE>
                                                                                              </PATH-EXPRESSION>
                                                                               </PACKAGE-REFERENCE>
                                                                               <PARAM-ASSIGNMENTS/>
                                                                               <ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="94da3a57-84c2-4ef5-83c8-9b3358e909b9" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Ausgangszustand pruefen (FwgSigIn1_Vmin, FwgSigIn1_VMax lesen)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="1275d4a7-68f6-4616-b331-b9b38e5f2ab4" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="timelessOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9dd6715c-2aa9-49a8-9727-835619406fc4" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator1MaxLimit_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">accelerator1MaxLimit</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <EXPECTATION xsi:type="timelessOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">4.5</VALUE>
                                                                                                              </VALUE>
                                                                                                              <TOLERANCE style="percentage" xsi:type="Tolerance">
                                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                                                              </VALUE>
                                                                                                              </TOLERANCE>
                                                                                              </EXPRESSION>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_v</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="0d3602d6-0eda-4aae-b32b-49fe698b532c" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="6371bc6e-6069-4861-8dd7-d320c32ab577" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">accelerator2MaxLimit</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_v</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="727ecd69-d3d1-4abc-86b6-f610108b2997" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">maxGradient</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_undefined_perc/s</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="38c035c9-495c-4bfb-8c5a-1f990f077b3a" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">fwg2_v</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="d417c502-af59-42bc-9502-5cef1f9cd89a" name="TsStartTrace" xsi:type="utility-9308d99e-50ac-11dc-8fe5-001143176a1c">
                                                                               <NAME xsi:type="string">Reaktionszeit</NAME>
                                                                               <RECORDING-GROUP-REF-BY-UUID>6170fd80a38411eb9ac4c8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="7aeae68e-36f6-452b-8f27-8a601e34d37f" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">50</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="f21d1dea-a276-46a8-b3a2-641a2a5911f3" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern  (FwgSigIn1 &gt; FwgSigIn1_VMax setzen)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="87dcb11c-34b3-4527-9ce0-0ab2b9a97c13" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="11e1e13a-6a57-46bd-a76e-6d5b89aba6cf" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.2</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                               </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="4c407482-c988-4422-98e7-2b074b0902cd" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion pruefen (Safe state: &quot;invalid&quot; signal FwgSig1_v)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="510327a2-6114-40df-82f2-89462d4877ba" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">1.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">500</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="3115e0a0-d4fa-4d3d-93ed-69a7ef4aae37" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">pctAccRef</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9d3e4a26-d77e-4626-a99f-5e8aadf0f10e" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">fwg2_v</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="08a18a7c-7f39-4dbf-9f98-12bf8cdc92aa" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">50</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="7779f48b-8953-4e34-b5e8-c16dbd328824" name="TsStopTrace" xsi:type="utility-d12f5791-50ac-11dc-8eb4-001143176a1c">
                                                                               <NAME xsi:type="string">Reaktionszeit</NAME>
                                                                               <RECORDING-GROUP-REF-BY-UUID>6170fd80a38411eb9ac4c8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="6f6d3849-61e5-4416-bf00-0bc5ad8afba8" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern (FWG2 manipulieren)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="94be66aa-0240-44b8-a214-6c36bdf40e0f" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="20233ec1-3386-4bca-8320-b8af55351bac" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">fwg2_v</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.2</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="07391d6e-c9aa-49ca-ac70-8f870b4760f3" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion pruefen (PctAcceleration folgt FWG2)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="7f3aa7ca-8b89-40b1-a545-b13f728667ca" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">1.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">500</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="f2921e7f-8d22-4446-9e02-0312e78d2e88" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">&gt;=</RELATION>
                                                                                                              <VALUE xsi:type="binaryOpBaseExpression">
                                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                                              <NAME xsi:type="string">pctAccRef</NAME>
                                                                                                                              </FIRST-COMPONENT>
                                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                                              <VALUE xsi:type="integer">2</VALUE>
                                                                                                                              </SECOND-COMPONENT>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1000</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="c9578c6b-4c6f-46ec-8dff-1b93acf1edfe" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern (FWG2 Max Wert zur Gradientenbestimmung)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                               <TESTSTEP id="ff968496-8dcd-41e6-bf42-57af6cedffa8" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_SUBTRACT</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.1</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="c21ce869-3253-4af6-b067-066e79b9f1fa" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion prüfen (Gradientenmessung per Traceanalyse)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                               <TESTSTEP id="5c459f74-0dde-45e7-8de9-c783783c16f8" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">1000</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="735fe1b1-7860-4c13-9c6b-db86d1d85793" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zuruecksetzen</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="93ab0edd-28a0-4f19-a8cb-4bdc198e4adc" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">0</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="babae8a8-576f-40a7-b2f1-d3ad0367fd28" xsi:type="tsRestore">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9b075b78-d6a1-4a15-8b32-b047838afa6b" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">0</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="a4404e8f-e2ed-463f-930d-dafcd2cc5b92" xsi:type="tsRestore">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="2d70af9b-6c3e-4071-a963-b6a9327938ad" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Postcondition</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP format-rev="3" id="750af70a-b500-4b25-a1ee-2fa83fac9ec1" xsi:type="tsPackage">
                                                                               <PACKAGE-REFERENCE xsi:type="expressionref">
                                                                                              <PATH-EXPRESSION format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="string">C:\\HIL_AE\\Packages\\Packages\\lib\\TerminateExecution.pkg</VALUE>
                                                                                              </PATH-EXPRESSION>
                                                                               </PACKAGE-REFERENCE>
                                                                               <PARAM-ASSIGNMENTS>
                                                                                              <ASSIGNMENT dkey="Fahrzeug_stillstand">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Fehlerspeicher">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Klemmenwechsel_KL15">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">False</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Klemmenwechsel_KL30">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                               </PARAM-ASSIGNMENTS>
                                                                               <ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <MAPPING format-rev="1" xsi:type="temporaryMappingSpace">
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.PctAccelerator</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_PctAcceleratorFailPositiveGradient_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_UAccelerator1MaxLimit_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_UAccelerator1MaxLimit_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_UAccelerator2MaxLimit_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Active/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Control/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Active/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Control/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                               </MAPPING>
                               </ELEMENT>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">maxGradient</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">maxGradient</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list"/>
                </ITEMS>
</EIX-LIST>
"""
y = """<?xml version="1.0" encoding="utf-8"?>
<PACKAGE format-rev="7" prog-version="2020.2.98502" xmlns="http://www.tracetronic.de/xml/ecu-test" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.tracetronic.de/xml/ecu-test package.xsd">
	<INFORMATION format-rev="2" xsi:type="packageInfo">
		<ATTRIBUTES>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Type</NAME>
				<VALUE xsi:type="string">MANUAL</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Path</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Creation Date</NAME>
				<VALUE xsi:type="string">2020-06-04</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Status</NAME>
				<VALUE xsi:type="string">Design</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Estimated DevTime</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Execution Priority</NAME>
				<VALUE xsi:type="string">01-mandatory to be conducted</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Test Execution</NAME>
				<VALUE xsi:type="string">manual</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">KIFA Bereich</NAME>
				<VALUE xsi:type="string">ALL</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Designer</NAME>
				<VALUE xsi:type="string">qxi0247</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">TestInstance</NAME>
				<VALUE xsi:type="string">HIL_TSP_Antrieb_01</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Individ. Field 1</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Model involved</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">OMR-TA</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">ECU</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Priority</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Regression relevant</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Design Organization</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Estimated Duration (min)</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
		</ATTRIBUTES>
		<DESCRIPTION xsi:type="string">TSC_216:
The AE shall check that the FwgSigIn1 signal is within the valid range of [FwgSigIn1_Vmin, FwgSigIn1_VMax]  and mark it as &quot;invalid&quot; if the signal is out of range.

FTT &lt; 50 ms
Safe state: &quot;invalid&quot; signal FwgSig1_v

TSC_232:
The AE shall use the value of the FwgSig2_v as reference for the accelerator signal AccPos and limit the positive gradient of AccPos signal to a maximum AccPos_MaxGrad  if the FwgSig1_v is marked as invalid and the FwgSig2_v is marked as valid and the BrkAbs_v is not set.

Note: The controllability is only dependent on the FwgSig1_v and FwgSig2_v signals condition, the BrkAbs_v signal condition is only considered for comfort purposes.

FTT &lt; 50 ms
Safe state: Signal AccPos = FwgSig2_v and positive gradient limited to AccPos_MaxGrad  [%/s] and mark AccPos signal as &quot;fault&quot;

Zum Test wird:
 * Das applizierte Limit ermittelt und die FWG1 Spannung um 0.2V darunter gesetzt.
 * Die Reaktion von  RecuProhibitionAccelerator überwacht.
 * Die Reaktionszeit &lt; 50ms mittels Traceanalyse verifiziert.
 * Der Gradient &lt; 400% mittels Traceanalyse verifiziert

Erstellt: 26.05.2021 (Ralf Zeischka)</DESCRIPTION>
		<VERSION xsi:type="string">$Rev: 121 $</VERSION>
		<TAGS>
			<TAG xsi:type="string">TESTCASE</TAG>
		</TAGS>
		<ALTERNATE-CALL-REPRESENTATION-ACTION-FIELD-TEMPLATE xsi:type="string"/>
		<ALTERNATE-CALL-REPRESENTATION-EXPECTATION-FIELD-TEMPLATE xsi:type="string"/>
	</INFORMATION>
	<VARIABLES xsi:type="variableContainer">
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator2MaxLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">maxGradient</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator1MinLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">fwg2_v</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAccRef</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAcceleratorWert1</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAcceleratorWert2</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator2MinLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">trigg</NAME>
			<DESCRIPTION xsi:type="string"/>
			<DEFAULT-VALUE format-rev="1" xsi:type="value">
				<DATA xsi:type="integer">0</DATA>
				<TEXTDATA xsi:type="string">0</TEXTDATA>
			</DEFAULT-VALUE>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">startGradientenmessung</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">startGradientenmessung1</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
	</VARIABLES>
	<MAPPING format-rev="1" xsi:type="localMappingSpace">
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_ElectricDriveReceiver.PctAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.PctAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_PctAcceleratorFailPositiveGradient_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator1MinLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator1MinLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator2MaxLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MinLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator2MinLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">Fahrbereitschaft</ID>
			<XACCESS format-rev="2" xsi:type="xaModelSignal">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/SGVM_MOTORRAD_AE/KOMBI/TA_BMS_AE_Fahrbereitschaft/Switch/Out1</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">Killswitch</ID>
			<DESCRIPTION xsi:type="string">Killswitch E_S_KILL; Ecu Pin 25; BOB B02</DESCRIPTION>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/INTERFACE_IO_BMS_AE/CTRL_IO_BMS/Killswitch_[0|1]/CTRL_E_S_KILL_[0|1]/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">StartTaster</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/INTERFACE_IO_BMS_AE/CTRL_IO_BMS_AE/CTRL_E_S_START_[0|1]/CTRL_E_S_START_[0|1]/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_BMS_AE_Fahrbereitschaft</ID>
			<XACCESS format-rev="2" xsi:type="xaModelSignal">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/SGVM_MOTORRAD_AE/KOMBI/TA_BMS_AE_Fahrbereitschaft/Switch/Out1</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Active/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Control/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Active/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Control/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
	</MAPPING>
	<TM-INFO format-rev="1" xsi:type="testManagementInfo"/>
	<TRACE-RECORDING format-rev="2" xsi:type="recordingManager">
		<SIGNAL-GROUPS>
			<ELEMENT format-rev="2" xsi:type="signalGroup">
				<NAME xsi:type="string">Gradientenmessung 1</NAME>
				<DESCRIPTION xsi:type="string"/>
				<SIGNALS>
					<ELEMENT xsi:type="signal">
						<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</MAPPING-REF>
					</ELEMENT>
				</SIGNALS>
			</ELEMENT>
		</SIGNAL-GROUPS>
		<RECORDING-GROUP>
			<ELEMENT format-rev="3" xsi:type="recordingGroup">
				<NAME xsi:type="string">Aufnahmegruppe von Gradientenmessung 1</NAME>
				<UUID xsi:type="string">31d6ef18a15a11eb849bc8d9d205aa05</UUID>
				<SIGNAL-GROUP-REF>Gradientenmessung 1</SIGNAL-GROUP-REF>
			</ELEMENT>
		</RECORDING-GROUP>
		<SYNCHRONISATION-CONFIG xsi:type="syncConfig"/>
	</TRACE-RECORDING>
	<TESTSTEPS xsi:type="testCase">
		<TESTSTEP format-rev="3" id="6543b0e9-2e1b-427a-a88f-65d2d9486fa4" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Precondition</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP format-rev="2" id="27995146-601b-4152-bfcc-1ca80dcf0f77" xsi:type="tsPackage">
				<PACKAGE-REFERENCE format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="string">..\\..\\lib\\HiL Init.pkg</VALUE>
				</PACKAGE-REFERENCE>
				<PARAM-ASSIGNMENTS/>
				<ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="94da3a57-84c2-4ef5-83c8-9b3358e909b9" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Ausgangszustand pruefen (FwgSigIn1_Vmin, FwgSigIn1_VMax lesen)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="8d1a2e0d-8ed4-4d19-bb05-7e3ddd4adcf2" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.0</VALUE>
						</VALUE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="4fcd71fb-3649-435f-83c7-09c321dc48d7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator1MinLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator1MinLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.7</VALUE>
						</VALUE>
						<TOLERANCE style="percentage" xsi:type="Tolerance">
							<VALUE format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="integer">1</VALUE>
							</VALUE>
						</TOLERANCE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="3c42a7f4-a44f-45b3-a29b-1e58e75073d0" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator2MaxLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="4daf152f-7bc8-423a-a060-a3f708aa46f7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MinLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator2MinLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.3</VALUE>
						</VALUE>
						<TOLERANCE style="absolute-value" xsi:type="Tolerance">
							<VALUE format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="float">0.01</VALUE>
							</VALUE>
						</TOLERANCE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="ad6ec1c3-f4e1-4062-9a76-7793bc5ac57a" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">maxGradient</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_undefined_perc/s</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="9222e164-15da-490e-9b7e-f8c45ba6efe7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">fwg2_v</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="1c982b9b-f376-4b21-a169-964603d516ac" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="2191552f-c313-4073-9779-c227cdfb7341" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern  (FwgSigIn1 &lt; FwgSigIn1_VMin setzen)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="8e661123-05d6-40dc-ba61-2a73523f15e2" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">1</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="24292a7b-03fe-4612-884a-ed9107c7423e" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">accelerator1MinLimit</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.2</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="4c09ee91-fe69-499c-ab57-c8329a9d84e8" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion pruefen (Safe state: &quot;invalid&quot; signal FwgSig1_v)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="a3498426-8406-4a6d-b6ef-95ecc465766e" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">1.0</VALUE>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="a7dc905b-3809-4f0e-881a-15cf583029d8" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">pctAccRef</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="71108f35-5f48-4de5-a74a-4d9ce6e163a8" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">fwg2_v</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="e31e741a-0e42-4cd3-a74a-d1938b3a9a81" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="10f2dbd2-afd6-4447-9ee7-960573211274" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern (FWG2 manipulieren)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="0d7f02f8-3a46-4677-86be-b8626587b16f" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">1</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="593bbac2-bcd2-4f1e-8e5b-cf5b0af710f5" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_ADD</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">fwg2_v</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.2</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="2f2e0c03-6bb9-428b-9538-14174b909b12" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">500</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="6ea0cb9d-d1f3-43b9-a22a-2758f3f068c1" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion pruefen (PctAcceleration folgt FWG2)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="f292540a-b0e2-4cd2-a8a5-6416f49d8d59" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">1.0</VALUE>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="18fc5c8f-3614-4cec-906b-d47d89ff152f" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">&gt;=</RELATION>
						<VALUE xsi:type="binaryOpBaseExpression">
							<NAME xsi:type="string">BINARY_ADD</NAME>
							<FIRST-COMPONENT xsi:type="varBaseExpression">
								<NAME xsi:type="string">pctAccRef</NAME>
							</FIRST-COMPONENT>
							<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="integer">2</VALUE>
							</SECOND-COMPONENT>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="ffd6106d-c36e-4ee6-bc59-46d752c91809" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern (FWG2 Max Wert zur Gradientenbestimmung)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="e963700f-2941-422f-b29e-546f81143876" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">accelerator2MaxLimit</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.1</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="8619f375-7ee0-4f32-b2a9-b0eb54545ab2" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="e96ac323-9a18-442b-b14f-d74d0db4e85b" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion prüfen (Gradientenmessung per Traceanalyse)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="86ff1301-ed56-40ea-b054-a9ba24c0409e" name="TsStartTrace" xsi:type="utility-9308d99e-50ac-11dc-8fe5-001143176a1c">
				<NAME xsi:type="string">Gradientenmessung</NAME>
				<RECORDING-GROUP-REF-BY-UUID>31d6ef18a15a11eb849bc8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
			</TESTSTEP>
			<TESTSTEP id="7afd0f0b-f0e1-4137-a2a7-0ca95957fbb2" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">250</VALUE>
				</TIME>
			</TESTSTEP>
			<TESTSTEP id="c1a71aa1-f70f-498b-af0f-dc68fc42e0a4" name="TsStopTrace" xsi:type="utility-d12f5791-50ac-11dc-8eb4-001143176a1c">
				<NAME xsi:type="string">Gradientenmessung</NAME>
				<RECORDING-GROUP-REF-BY-UUID>31d6ef18a15a11eb849bc8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="735fe1b1-7860-4c13-9c6b-db86d1d85793" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zuruecksetzen</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="f8403c81-9cae-4e33-9fe8-5956e4ec1920" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="1fef7048-a519-437e-8d08-238ce22729e2" xsi:type="tsRestore">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
			</TESTSTEP>
			<TESTSTEP id="9e421a84-a888-4efb-ba8b-03841e990955" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator1MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="b48a4a42-e09d-4055-b9d6-1605e11dfaf8" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="bad88484-dcc8-4f11-9db0-3c72e4c69bd1" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="float">0.0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="cdb13ac1-c3ab-438f-bd4f-7887835b891a" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="float">0.0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="810076c3-0c38-40f0-9bb4-78a6c87d1aec" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator1MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="e5bd43d1-216f-4fa6-9c1e-def807257043" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator2MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="326a0c85-3825-4a95-9d2b-f8447b454835" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Postcondition</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP format-rev="2" id="6b7f756b-ac87-4ad4-bca4-3b6a69452d61" xsi:type="tsPackage">
				<PACKAGE-REFERENCE format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="string">C:\\HIL_AE\\Packages\\Packages\\lib\\TerminateExecution.pkg</VALUE>
				</PACKAGE-REFERENCE>
				<PARAM-ASSIGNMENTS>
					<ASSIGNMENT dkey="Fahrzeug_stillstand">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Fehlerspeicher">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Klemmenwechsel_KL15">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">False</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Klemmenwechsel_KL30">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
				</PARAM-ASSIGNMENTS>
				<ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
			</TESTSTEP>
		</TESTSTEP>
	</TESTSTEPS>
	<TRACE-ANALYSES format-rev="1" xsi:type="traceAnalyses">
		<TRACE-ANALYSIS format-rev="3" uuid="0f91da22a14c11eb9cf5c8d9d205aa05" xsi:type="traceAnalysis">
			<NAME xsi:type="string">Gradientenmessung</NAME>
			<ANALYSISITEM format-rev="1" uuid="0f92014ca14c11ebbf92c8d9d205aa05" xsi:type="episode">
				<NAME xsi:type="string">Neue Episode</NAME>
				<ANALYSISITEM format-rev="2" poolRev="6" uuid="983e2637a3a211eb9607c8d9d205aa05" xsi:type="referencingTraceStep">
					<NAME xsi:type="string">Gradient_Zeitfenster</NAME>
					<SIGNAL-BINDING xsi:type="signalBinding">
						<PARAMETER dkey="messwert">
							<DVALUE format-rev="1" xsi:type="signalParameter">
								<REF-SIGNAL-NAME xsi:type="string">pctAccelerator</REF-SIGNAL-NAME>
							</DVALUE>
						</PARAMETER>
					</SIGNAL-BINDING>
					<PARAM-ASSIGNMENTS xsi:type="trcpParameterAssignments">
						<PARAM-ASSIGNMENTS>
							<ASSIGNMENT dkey="MaxGradient">
								<DVALUE xsi:type="binaryOpBaseExpression">
									<NAME xsi:type="string">BINARY_ADD</NAME>
									<FIRST-COMPONENT xsi:type="varBaseExpression">
										<NAME xsi:type="string">maxGradient</NAME>
									</FIRST-COMPONENT>
									<SECOND-COMPONENT xsi:type="binaryOpBaseExpression">
										<NAME xsi:type="string">BINARY_MULTIPLY</NAME>
										<FIRST-COMPONENT xsi:type="varBaseExpression">
											<NAME xsi:type="string">maxGradient</NAME>
										</FIRST-COMPONENT>
										<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
											<VALUE xsi:type="float">0.05</VALUE>
										</SECOND-COMPONENT>
									</SECOND-COMPONENT>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="MinGradient">
								<DVALUE xsi:type="binaryOpBaseExpression">
									<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
									<FIRST-COMPONENT xsi:type="varBaseExpression">
										<NAME xsi:type="string">maxGradient</NAME>
									</FIRST-COMPONENT>
									<SECOND-COMPONENT xsi:type="binaryOpBaseExpression">
										<NAME xsi:type="string">BINARY_MULTIPLY</NAME>
										<FIRST-COMPONENT xsi:type="varBaseExpression">
											<NAME xsi:type="string">maxGradient</NAME>
										</FIRST-COMPONENT>
										<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
											<VALUE xsi:type="float">0.05</VALUE>
										</SECOND-COMPONENT>
									</SECOND-COMPONENT>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="Startzeit">
								<DVALUE format-rev="1" xsi:type="valueBaseExpression">
									<VALUE xsi:type="integer">0</VALUE>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="Stopzeit">
								<DVALUE format-rev="1" xsi:type="valueBaseExpression">
									<VALUE xsi:type="integer">500</VALUE>
								</DVALUE>
							</ASSIGNMENT>
						</PARAM-ASSIGNMENTS>
					</PARAM-ASSIGNMENTS>
					<REPORT-CONFIG xsi:type="trcpReportConfig">
						<MAX-SPOTS xsi:type="integer">100</MAX-SPOTS>
						<MIN-DELTA xsi:type="float">0.0</MIN-DELTA>
					</REPORT-CONFIG>
					<PROTOTYPE-ID xsi:type="string">Gradient_Zeitfenster</PROTOTYPE-ID>
				</ANALYSISITEM>
			</ANALYSISITEM>
			<VIRTUAL-SIGNALS-MANAGER xsi:type="virtualSignalsManager">
				<VIRTUAL-SIGNALS>
					<VIRTUAL-SIGNAL uuid="5cef80faa14c11eb983dc8d9d205aa05" xsi:type="virtualSessionSignal">
						<NAME xsi:type="string">pctAccelerator</NAME>
						<SIGNAL-FOR-ANALYSIS xsi:type="recordingSignalForAnalysis">
							<KEY xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</KEY>
							<SOURCE-TYPE xsi:type="string">RECORDING-GROUP</SOURCE-TYPE>
							<SOURCE-REF xsi:type="string">31d6ef18a15a11eb849bc8d9d205aa05</SOURCE-REF>
						</SIGNAL-FOR-ANALYSIS>
						<EVENT-TYPE xsi:type="string">PHY</EVENT-TYPE>
					</VIRTUAL-SIGNAL>
				</VIRTUAL-SIGNALS>
			</VIRTUAL-SIGNALS-MANAGER>
		</TRACE-ANALYSIS>
	</TRACE-ANALYSES>
</PACKAGE>
"""


class ExcelStyle:

    yellow_fill = PatternFill(start_color='FFFFCC', end_color='FFFFCC', fill_type='solid')

    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))
    merge_a_to_c = lambda xx: MergeCells(start_row=xx, start_column=1, end_row=xx, end_column=4)


class ExcelRowBase:

    def __init__(self, columns: iter, style=None):
        self.columns = columns
        self.style = style

    # def __repr__(self):
    #     return f'{self.columns} {self.style}'

    def __str__(self):
        prs = self.style or ''
        try:
            vs = list(self.columns.values())
            vs = [x for x in vs if x is not None]
            vs = '\t'.join(vs)
        except TypeError as sfeh:
            vs = self.columns
        return f'{vs} \t\t{prs}'


class Excel_package:

    def __init__(self, name, description, rows=None):
        self.rows = [ExcelRowBase({2: name}, style='TODO'),
                     ExcelRowBase({2: description}, style='TODO')]

    def add_teststep(self, stepdict):
        dc = {1: stepdict['Aktion'],
              2: stepdict['Variablenname'],
              3: stepdict['Vorgabe/Erwartungswert']}
        style = None
        row = ExcelRowBase(dc, style)
        self.rows.append(row)

    def add_testblock(self, name):
        row = (ExcelRow_Teststep({2: name}, style=(ExcelStyle.yellow_fill, ExcelStyle.merge_a_to_c)))
        self.rows.append(row)


def test_triple(stepdict, style=None):
    dc = {1: stepdict['Aktion'],
          2: stepdict['Variablenname'],
          3: stepdict['Vorgabe/Erwartungswert']}
    style = None
    row = ExcelRowBase(dc, style)
    return row


class ExcelRow_Teststep(ExcelRowBase):
    def __init__(self, row, style=None):
        style = ExcelStyle.merge_a_to_c
        super().__init__(row, style=style)


class Excel_Sheet:

    def __init__(self, sheet_name):
        self.sheet_name = sheet_name
        self.rows = []

    def append(self, row:ExcelRowBase):
        self.rows.append(row)

    def extend(self, rows: iter('ExcelRowBase')):
        self.rows.extend(rows)


class ExcelRow_Testblock(ExcelRowBase):
    def __init__(self, description):
        style = ExcelStyle.merge_a_to_c
        super().__init__({2: description}, style=style)
        


def excel_row_border(row, style='thin'):
    row[0].border = Border(left=Side(style=style), top=Side(style=style), bottom=Side(style=style))
    row[-1].border = Border(right=Side(style=style), top=Side(style=style), bottom=Side(style=style))
    for cell in row[1:-1]:
        cell.border = Border(top=Side(style=style), bottom=Side(style=style))


def get_VALUE_expr(value):
    ex_type = value['@xsi:type']
    if ex_type == 'valueBaseExpression':
        b1 = value['VALUE']['@xsi:type']
        b2 = value['VALUE']['#text']
        b = f'{b2}'
    elif ex_type == 'expressionValue':
        data = value['DATA']
        b = get_VALUE_expr(data)
    elif ex_type == 'varBaseExpression':
        b = value['NAME']['#text']

    elif ex_type == 'binaryOpBaseExpression':
        def get_component_string(c):
            if c['@xsi:type'] == 'varBaseExpression':
                return c['NAME']['#text']
            elif c['@xsi:type'] == 'valueBaseExpression':
                return c['VALUE']['#text']

        b1 = value['@xsi:type']
        b2 = value['NAME']['#text']
        b2 = re.sub('BINARY_ADD', '+', b2)
        b2 = re.sub('BINARY_SUB', '-', b2)
        b3 = get_component_string(value['FIRST-COMPONENT'])
        b4 = get_component_string(value['SECOND-COMPONENT'])
        b = f'({b3} {b2} {b4})'

    else:
        raise NotImplementedError
    return b


def get_TOLERANCE(expression):
    tolerance = expression.get('TOLERANCE')
    if tolerance is not None:
        a = tolerance['@style']  # percentage
        a = re.sub('percentage', '%', a)
        a = re.sub('absolute-value', '', a)
        b = tolerance['VALUE']['VALUE']['#text']
        s = f' [+-{b}{a}]'
    else:
        s = ''

    return s


def get_EXPRESSION(expectation):
    expression = expectation.get('EXPRESSION')
    if expression is not None:
        xsi_type = expression['@xsi:type']
        if xsi_type == 'builtNumericExpression':
            rel = expression['RELATION']['#text']
            value = expression['VALUE']
            b = get_VALUE_expr(value)
            tolerance = get_TOLERANCE(expression)
            s = f'{rel} {b}{tolerance}'
        else:
            raise NotImplementedError

        return s


def get_EXPECTATION(x):
    expectation = x.get('EXPECTATION')
    if expectation is not None:

        a = get_EXPRESSION(expectation)

        if expectation.get('@xsi:type') == 'timelessOption':
            s_time = ''
        elif expectation.get('@xsi:type') == 'finallyTrueOption':
            t0 = expectation['TIME']['@format-rev']
            t1 = expectation['TIME']['@xsi:type']
            # if t0 == '2' and t1 == 'valueBaseExpression':
            #     pass  # is this ms?
            # else:
            #     pass
            e_time = expectation['TIME']['VALUE']['#text']
            s_time = f'  [t: {e_time}ms]'
        else:
            raise NotImplementedError

        return f' {a}{s_time}'
    else:
        return f''


def get_SAVETO(x):
    s = x.get('VARIABLE-REFS')
    if s is not None:
        s = s['VARIABLE-NAME']['DVALUE']['#text']
        return f' -> {s}'
    else:
        return ''


def get_METRIC_expr(metric):
    try:
        s = metric['Z-UNIT']['#text']
        s = f' {s}'
    except KeyError as ex:
        s = ''
    s = re.sub('u_none', '', s)
    return s


def sanitize(ss):
    ss = re.sub('DME1_DDE1/', '', ss)
    ss = re.sub('/Control/Value', '', ss)
    ss = re.sub('/Active/Value', '', ss)
    return ss


IGNORE_USELESS_LEVEL = 1
# worksheet.set_column("A:A", 20)  # todo


def teststep_get_excel(ecutest):
    # todo check if is active

    if (ecutest.get('@xsi:type') or '') == 'noEventTestStepMappingContainer':
        return

    elif (ecutest.get('@xsi:type') or '') == 'testCase':
        return

    elif (ecutest.get('#text') or '') == '<padding>':
        return

    elif (ecutest.get('@name') or '') == 'TsBlock':  # @xsi:type is 'utility-2752ad1e-4fef-11dc-81d4-0013728784ee'
        bb = ecutest['ACTION']['MULTILANGDATA']['ELEMENT']['DVALUE']['#text']
        if IGNORE_USELESS_LEVEL >= 1 and ('Zuruecksetzen' in bb):
            return
        elif IGNORE_USELESS_LEVEL >= 2 and ('Precondition' in bb or 'Postcondition' in bb):
            return
        else:
            EXCEL_dict = test_triple({'Aktion': '', 'Variablenname': bb, 'Vorgabe/Erwartungswert': f''},
                                     style={'fill': None, 'row_border': None, 'is_teststep': True})

    elif (ecutest.get('@xsi:type') or '') == 'tsPackage':
        try:
            ecupath = ecutest['PACKAGE-REFERENCE']['VALUE']['#text']
        except KeyError:
            ecupath = ecutest['PACKAGE-REFERENCE']['PATH-EXPRESSION']['VALUE']['#text']

        ecupath = re.sub(r'.*\\', '', ecupath)

        if ecutest.get('PARAM-ASSIGNMENTS') is not None:
            params = ecutest['PARAM-ASSIGNMENTS']['ASSIGNMENT']
            b = []
            for p in params:
                p0 = p['@dkey']
                p1 = get_VALUE_expr(p['DVALUE'])
                b.append(f'{p0}={p1}')
            b = ', '.join(b)
            b = f' ({b})'
            if IGNORE_USELESS_LEVEL >= 2 and ('HIL Init.pkg' in ecupath or 'TerminateExecution.pkg' in ecupath):
                return
            elif IGNORE_USELESS_LEVEL >= 1 and ('HIL Init.pkg' in ecupath or 'TerminateExecution.pkg' in ecupath):
                b = ''
        else:
            b = ''
        EXCEL_dict = test_triple({'Aktion': 'Aufruf', 'Variablenname': ecupath, 'Vorgabe/Erwartungswert': f'{b}'},
                                 style={'fill': None, 'row_border': ExcelStyle.thin_border, 'is_teststep': True})

    elif ecutest.get('@xsi:type') == 'tsRead':
        ecupath = ecutest['MAPPING-REF']['#text']
        if IGNORE_USELESS_LEVEL >= 1 and '/Active/Value' in ecupath:
            return
        ecupath = sanitize(ecupath)
        save_to = get_SAVETO(ecutest)
        expectation = get_EXPECTATION(ecutest)
        s = f'{ecupath}{expectation}{save_to}'
        if expectation is not None:
            aa = 'prüfen'
        elif save_to is not None:
            aa = 'speichern'
        else:
            raise NotImplementedError('What is done here??')
        EXCEL_dict = test_triple({'Aktion': aa, 'Variablenname': ecupath, 'Vorgabe/Erwartungswert': f'{expectation}{save_to}'},
                                 style={'row_border': ExcelStyle.thin_border, 'is_teststep': True})

    elif ecutest.get('@xsi:type') == 'tsWrite':
        ecupath = ecutest['MAPPING-REF']['#text']
        if IGNORE_USELESS_LEVEL >= 1 and '/Active/Value' in ecupath:
            return
        ecupath = sanitize(ecupath)
        value = ecutest['VALUE']
        b = get_VALUE_expr(value)
        EXCEL_dict = test_triple({'Aktion': 'schreiben', 'Variablenname': ecupath, 'Vorgabe/Erwartungswert': f'{b}'},
                      style={'fill': None, 'row_border': ExcelStyle.thin_border, 'is_teststep': True})

    elif ecutest.get('@xsi:type') == 'tsRestore':
        # IGNORE_USELESS_LEVEL = 2
        return

    elif ecutest.get('@name') == 'TsWait':
        # s = ecutest['TIME']['VALUE']['#text']
        # s = f'--wait-- {s}'
        # IGNORE_USELESS_LEVEL = 3?
        return

    elif ecutest.get('@name') == 'TsStartTrace':
        s = ecutest['NAME']['#text']
        EXCEL_dict = test_triple({'Aktion': 'Traceanalyse (start)', 'Variablenname': s, 'Vorgabe/Erwartungswert': None},
                      style={'fill': None, 'row_border': ExcelStyle.thin_border, 'is_teststep': True})

    elif ecutest.get('@name') == 'TsStopTrace':
        s = ecutest['NAME']['#text']
        EXCEL_dict = test_triple({'Aktion': 'Traceanalyse (stop)', 'Variablenname': f'{s}', 'Vorgabe/Erwartungswert': None},
                      style={'row_border': ExcelStyle.thin_border, 'is_teststep': True})

    elif ecutest.get('@xsi:type') == 'list':
        return  # 'DONE! (TODO check if this is last)'
    else:
        raise NotImplementedError

    return EXCEL_dict


def ecu_to_excel_recursive(ecu_xml):

    if ecu_xml.get('ENABLED') or False:
        # print(f'Disabled!')
        return
    else:
        try:
            ecu_xml = ecu_xml['TESTSTEPS']
        except KeyError:
            pass

        teststep_xml = ecu_xml.get('TESTSTEP') or None

        if teststep_xml is None:  # None, if no teststeps
            excel_teststep = teststep_get_excel(ecu_xml)
            return [excel_teststep]

        else:
            if isinstance(teststep_xml, dict):  # dict, if only one teststep
                excel_teststep = teststep_get_excel(teststep_xml)
                return [excel_teststep]

            elif isinstance(teststep_xml, list):  # list, if multiple teststep
                try:
                    excel_list = [teststep_get_excel(ecu_xml)]
                except Exception as todo:
                    excel_list = []
                for tt in teststep_xml:
                    tn = ecu_to_excel_recursive(tt)
                    if isinstance(tn, list) and len(tn) > 0:
                        excel_list.extend(tn)
                    elif isinstance(tn, ExcelRowBase):
                        excel_list.append(tn)
                    elif tn is None:
                        pass  # todo
                    else:
                        raise NotImplementedError
                # excel_list = [x[0] if isinstance(x, list) else x for x in excel_list]
                excel_list = [x for x in excel_list if x is not None]
                return excel_list
            else:
                raise NotImplementedError


def xls_to_excel_writer(rows):
    excel_path = 'example.xlsx'
    excel_sheet = 'Sheet1'
    startrow = 0

    # DataFrame in eine Excel-Datei speichern
    # Die Excel-Datei mit openpyxl laden
    workbook = Workbook(excel_path)
    # wb = openpyxl.load_workbook(excel_path)
    ws = workbook.create_sheet(excel_sheet, 0)
    print(workbook.sheetnames)
    workbook.active = workbook[excel_sheet]
    ws = workbook.active

    # Format all the columns.
    my_format = workbook.add_format()
    my_format.set_text_wrap()

    ws.set_column('A:D', None, my_format)

    # # Überprüfen und Zeilen färben, wenn in Reihe 2 der Text "Block" steht
    # for row in ws.iter_rows(min_row=0, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
    #     print(f'{row[0].value}: {row[0]}')
    #     if row[0].value is None:  # Block, Index 1 bezieht sich auf die zweite Spalte (B)
    #         for cell in row:
    #             cell.fill = ExcelStyle.yellow_fill
    #         excel_row_border(row)
    for row in rows:
        vs = list(row.columns.values())
        style = row.style
        ws.append(vs)

    # Änderungen speichern
    workbook.save(excel_path)
    print(f"Excel-Datei wurde erfolgreich unter {excel_path} gespeichert und formatiert.")


# def xls_to_excel_writer
#
#     with pd.ExcelWriter(excel_path) as writer:
#         for row in rows:
#             row.ecu_to_excel_recursive(writer, engine="xlsxwriter", sheet_name='Sheet1', startrow=startrow, index=False)
#             startrow += (.shape[0] + 3)

PATH_PROJECTS = Path('C:/Users/Simon/Documents/BMW-Motorrad/#Oktober/HIL_AE/Packages/Projects/')
PATH_PACKAGES = Path('C:/Users/Simon/Documents/BMW-Motorrad/#Oktober/HIL_AE/Packages/')


def proj_get_testcase(testcase):
    if testcase['ENABLED']['#text'] == 'True':
        if testcase['@xsi:type'] == 'subproject':
            raise NotImplementedError
        elif testcase['@xsi:type'] == 'packageTest':
            if testcase['ENABLED']['#text'] == 'True':
                p_testcase = PATH_PROJECTS / testcase['PACKAGE-REF']['PACKAGE-PATH']['#text']  # creates actual path
                if not p_testcase.exists():
                    raise FileNotFoundError(f'Testcase not found: {p_testcase}')
                return p_testcase
        else:
            raise NotImplementedError
    else:
        return None


def get_testcase_dict():
    # C:\Users\Simon\Documents\BMW-Motorrad\#Oktober\HIL_AE\Packages\Projects\AlleDTCs.prj
    # AlleDTCs
    sheet_cases_dict = {}
    with Path.open('C:/Users/Simon/Documents/BMW-Motorrad/#Oktober/HIL_AE/Packages/Projects/AlleDTCs.prj', 'r') as file:
        data = file.read()
        proj = xmltodict.parse(data)
    subname = proj['PROJECT']['COMPONENTS']['COMPONENT'][3]['NAME']['#text']

    sheet_cases_dict['OBD_DTCs_BMW'] = []
    for tt in proj['PROJECT']['COMPONENTS']['COMPONENT'][3]['COMPONENTS']['COMPONENT'][0]['COMPONENTS']['COMPONENT']:
        sheet_cases_dict['OBD_DTCs_BMW'].append(proj_get_testcase(tt))

    sheet_cases_dict['OBD_DTCs_Marelli'] = []
    for tt in proj['PROJECT']['COMPONENTS']['COMPONENT'][3]['COMPONENTS']['COMPONENT'][1]['COMPONENTS']['COMPONENT']:
        sheet_cases_dict['OBD_DTCs_Marelli'].append(proj_get_testcase(tt))

    sheet_cases_dict['Other'] = []
    for tt in proj['PROJECT']['COMPONENTS']['COMPONENT'][3]['COMPONENTS']['COMPONENT'][2:]:
        # name = tt['NAME']['#text']
        sheet_cases_dict['Other'].append(proj_get_testcase(tt))

    return sheet_cases_dict


hohoho = xmltodict.parse(x)

TESTCASE_INIT = 1


def xls_from_package_file(file: Path):
    ecu_testcase_xml = xmltodict.parse(y)
    try:
        p_desc = ecu_testcase_xml['PACKAGE']['INFORMATION']['DESCRIPTION']['#text']
        # p_desc = p_desc.replace('\n', 'multiline\\015string')  # todo 'multiline\015string' is excel newline
    except KeyError:
        p_desc = None

    if file is not None:
        xls_rows = [ExcelRowBase({2: file.name}, style='TODO'),
                    ExcelRowBase({2: p_desc}, style={'alignment': Alignment(wrapText=True)}),
                    ExcelRowBase({1: 'Aktion', 2: 'Variablenname', 3: 'Vorgabe/Erwartungswert'}, style=['bold', 'thick'])]

        testpkg = ecu_testcase_xml['PACKAGE']
        xls_subrows = ecu_to_excel_recursive(testpkg)
        xls_rows.extend(xls_subrows)

        placeholder_todo = ExcelRowBase({0: ''})
        placeholder1_todo = ExcelRowBase({0: '====================================================================='})
        xls_rows.append(placeholder1_todo)  # just clear some lines
        xls_rows.append(placeholder_todo)  # just clear some lines
        return xls_rows
    else:
        placeholder1_todo = ExcelRowBase({0: '====================================================================='})
        return [placeholder1_todo]  # todo


subproj_packlist_dict = get_testcase_dict()
xlssheets_xlsdict = []
for package_p in subproj_packlist_dict['OBD_DTCs_BMW']:
    xls_rows = xls_from_package_file(package_p)
    print(f'\n\n')
    for prnt in xls_rows:
        print(prnt)
    xlssheets_xlsdict.extend(xls_rows)
xls_to_excel_writer(xlssheets_xlsdict)

