"""
Document Identification Validation and Testing
Comprehensive validation logic for the enhanced document identification system
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .enhanced_document_identification import DocumentIdentification, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of document identification validation"""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    completeness_score: float


class DocumentValidator:
    """Validates document identification results"""

    def __init__(self):
        self.federal_requirements = self._load_federal_requirements()
        self.validation_rules = self._load_validation_rules()

    def _load_federal_requirements(self) -> Dict[str, Any]:
        """Load federal contracting requirements"""
        return {
            "mandatory_documents": {
                "technical_rfp": ["technical_volume", "cost_volume"],
                "full_rfp": ["technical_volume", "cost_volume", "past_performance"],
                "rfi": ["capability_statement"],
                "rfq": ["price_narrative"],
            },
            "volume_conventions": {
                "dod": ["technical", "cost", "management", "past_performance"],
                "civilian": ["technical", "cost", "past_performance"],
                "nasa": ["technical", "cost", "management", "past_performance", "safety"]
            },
            "format_requirements": {
                "technical_volume": ["docx", "pdf"],
                "cost_volume": ["xlsx", "pdf"],
                "video_pitch": ["mp4", "script"]
            }
        }

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        return {
            "minimum_confidence": 0.3,
            "recommended_confidence": 0.7,
            "maximum_documents": 15,
            "required_document_types": ["technical_volume", "cost_volume"],
            "format_validation": True,
            "naming_convention_check": True
        }

    async def validate_document_identification(
        self,
        documents: List[DocumentIdentification],
        solicitation_type: str = "rfp",
        agency: str = "unknown"
    ) -> ValidationResult:
        """Comprehensive validation of document identification"""
        try:
            logger.info(f"Validating {len(documents)} identified documents for {solicitation_type}")

            issues = []
            warnings = []
            recommendations = []
            total_confidence = 0.0

            # Basic validation checks
            if not documents:
                issues.append("No documents identified")
                return ValidationResult(False, 0.0, issues, warnings, recommendations, 0.0)

            # Check document count
            if len(documents) > self.validation_rules["maximum_documents"]:
                warnings.append(f"High document count ({len(documents)}), consider consolidation")

            # Validate each document
            valid_documents = 0
            for doc in documents:
                doc_validation = await self._validate_single_document(doc, solicitation_type)
                total_confidence += doc.confidence_score

                if doc_validation["is_valid"]:
                    valid_documents += 1
                else:
                    issues.extend(doc_validation["issues"])

                warnings.extend(doc_validation["warnings"])
                recommendations.extend(doc_validation["recommendations"])

            # Check for required document types
            document_types = {doc.document_type for doc in documents}
            required_types = self._get_required_types(solicitation_type)

            for required_type in required_types:
                if required_type not in document_types:
                    issues.append(f"Missing required document type: {required_type.value}")

            # Check for duplicates
            document_names = [doc.name for doc in documents]
            duplicates = set([name for name in document_names if document_names.count(name) > 1])
            if duplicates:
                issues.append(f"Duplicate document names: {duplicates}")

            # Calculate scores
            avg_confidence = total_confidence / len(documents) if documents else 0.0
            completeness_score = (valid_documents / len(documents)) * 100 if documents else 0.0

            # Overall validation
            is_valid = (
                len(issues) == 0 and
                avg_confidence >= self.validation_rules["minimum_confidence"] and
                completeness_score >= 80.0
            )

            # Generate recommendations
            if avg_confidence < self.validation_rules["recommended_confidence"]:
                recommendations.append("Consider human review due to low confidence scores")

            if completeness_score < 90.0:
                recommendations.append("Review document requirements for completeness")

            logger.info(f"Validation complete: {is_valid}, confidence: {avg_confidence:.2f}, completeness: {completeness_score:.1f}%")

            return ValidationResult(
                is_valid=is_valid,
                confidence_score=avg_confidence,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                completeness_score=completeness_score
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(False, 0.0, [f"Validation error: {e}"], [], [], 0.0)

    async def _validate_single_document(self, doc: DocumentIdentification, solicitation_type: str) -> Dict[str, Any]:
        """Validate a single document"""
        issues = []
        warnings = []
        recommendations = []

        # Confidence check
        if doc.confidence_score < self.validation_rules["minimum_confidence"]:
            issues.append(f"Low confidence score for {doc.name}: {doc.confidence_score:.2f}")

        # Format validation
        expected_formats = self.federal_requirements["format_requirements"].get(
            doc.document_type.value, ["docx", "pdf"]
        )
        if doc.format not in expected_formats:
            warnings.append(f"Unusual format for {doc.name}: {doc.format} (expected: {expected_formats})")

        # Naming convention check
        if not self._check_naming_convention(doc.name):
            warnings.append(f"Non-standard naming convention: {doc.name}")

        # Requirements completeness
        if len(doc.requirements) < 50:  # Arbitrary threshold
            recommendations.append(f"Consider expanding requirements for {doc.name}")

        # Assumptions check
        if len(doc.assumptions) > 3:
            warnings.append(f"High number of assumptions for {doc.name}: {len(doc.assumptions)}")

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations
        }

    def _get_required_types(self, solicitation_type: str) -> List[DocumentType]:
        """Get required document types for solicitation"""
        type_mapping = {
            "rfp": [DocumentType.TECHNICAL_VOLUME, DocumentType.COST_VOLUME],
            "rfi": [DocumentType.CAPABILITY_STATEMENT],
            "rfq": [DocumentType.PRICE_NARRATIVE],
            "sources_sought": [DocumentType.CAPABILITY_STATEMENT]
        }

        return type_mapping.get(solicitation_type.lower(), [])

    def _check_naming_convention(self, filename: str) -> bool:
        """Check if filename follows federal naming conventions"""
        # Basic checks for federal naming conventions
        if any(char in filename for char in ['<', '>', ':', '"', '|', '?', '*']):
            return False

        # Should have extension
        if '.' not in filename:
            return False

        # Should not be too long
        if len(filename) > 100:
            return False

        return True


class DocumentIdentificationTester:
    """Test the enhanced document identification system"""

    def __init__(self):
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for validation"""
        return [
            {
                "name": "Standard RFP with Volumes",
                "northstar": """
                SOLICITATION ANALYSIS: This is an RFP for IT Services

                SUBMISSION REQUIREMENTS:
                Volume I - Technical Proposal: Provide detailed technical approach
                Volume II - Cost Proposal: Provide detailed cost breakdown in Excel format
                Volume III - Management Proposal: Organizational structure and staffing
                Volume IV - Past Performance: Provide 3 relevant contract examples

                EVALUATION FACTORS:
                Factor 1: Technical Approach (40%)
                Factor 2: Cost/Price (30%)
                Factor 3: Management Approach (20%)
                Factor 4: Past Performance (10%)
                """,
                "expected_documents": [
                    {"type": "technical_volume", "name": "Volume_I_Technical_Proposal.docx"},
                    {"type": "cost_volume", "name": "Volume_II_Cost_Proposal.xlsx"},
                    {"type": "management_volume", "name": "Volume_III_Management_Proposal.docx"},
                    {"type": "past_performance", "name": "Volume_IV_Past_Performance.docx"}
                ]
            },
            {
                "name": "RFI with Capability Statement",
                "northstar": """
                SOLICITATION TYPE: Request for Information (RFI)

                SUBMISSION REQUIREMENTS:
                Submit a capability statement describing your company's qualifications
                and experience in cybersecurity services. Maximum 10 pages.

                Submit white paper on approach to zero trust architecture.
                """,
                "expected_documents": [
                    {"type": "capability_statement", "name": "Capability_Statement.docx"},
                    {"type": "white_paper", "name": "Zero_Trust_White_Paper.docx"}
                ]
            },
            {
                "name": "Complex RFP with Special Requirements",
                "northstar": """
                SOLICITATION: RFP for Software Development

                SUBMISSION REQUIREMENTS:
                1. Cover Letter (2 pages maximum)
                2. Executive Summary (5 pages maximum)
                3. Technical Proposal addressing SOW requirements
                4. Price Narrative explaining cost methodology
                5. Past Performance with CPARS ratings
                6. Video Pitch (5 minutes maximum) demonstrating solution
                7. Security Plan addressing NIST 800-171

                FORMAT: All documents in PDF except video (MP4)
                """,
                "expected_documents": [
                    {"type": "cover_letter", "name": "Cover_Letter.pdf"},
                    {"type": "executive_summary", "name": "Executive_Summary.pdf"},
                    {"type": "technical_proposal", "name": "Technical_Proposal.pdf"},
                    {"type": "price_narrative", "name": "Price_Narrative.pdf"},
                    {"type": "past_performance", "name": "Past_Performance.pdf"},
                    {"type": "video_pitch", "name": "Video_Pitch.mp4"},
                    {"type": "security_plan", "name": "Security_Plan.pdf"}
                ]
            }
        ]

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the enhanced system"""
        from .enhanced_document_identification import EnhancedDocumentIdentifier

        identifier = EnhancedDocumentIdentifier()
        validator = DocumentValidator()

        test_results = []

        for test_case in self.test_cases:
            logger.info(f"Running test: {test_case['name']}")

            try:
                # Run identification
                identified_docs = await identifier.identify_required_documents(
                    test_case["northstar"]
                )

                # Validate results
                validation_result = await validator.validate_document_identification(
                    identified_docs, "rfp"
                )

                # Compare with expected
                expected_types = {doc["type"] for doc in test_case["expected_documents"]}
                identified_types = {doc.document_type.value for doc in identified_docs}

                accuracy = len(expected_types.intersection(identified_types)) / len(expected_types) if expected_types else 0.0

                test_result = {
                    "test_name": test_case["name"],
                    "identified_count": len(identified_docs),
                    "expected_count": len(test_case["expected_documents"]),
                    "accuracy": accuracy,
                    "validation": validation_result,
                    "identified_types": list(identified_types),
                    "expected_types": list(expected_types),
                    "missing_types": list(expected_types - identified_types),
                    "extra_types": list(identified_types - expected_types)
                }

                test_results.append(test_result)

            except Exception as e:
                logger.error(f"Test failed: {test_case['name']}: {e}")
                test_results.append({
                    "test_name": test_case["name"],
                    "error": str(e),
                    "accuracy": 0.0
                })

        # Calculate overall results
        successful_tests = [r for r in test_results if "error" not in r]
        overall_accuracy = sum(r["accuracy"] for r in successful_tests) / len(successful_tests) if successful_tests else 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(test_results) - len(successful_tests),
            "test_results": test_results
        }


async def run_validation_demo():
    """Demonstration of the enhanced document identification system"""
    logger.info("Starting Enhanced Document Identification System Demo")

    tester = DocumentIdentificationTester()
    results = await tester.run_comprehensive_test()

    print("\n" + "="*80)
    print("ENHANCED DOCUMENT IDENTIFICATION SYSTEM - TEST RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print(f"Tests Passed: {results['successful_tests']}/{results['total_tests']}")
    print(f"Tests Failed: {results['failed_tests']}")

    print("\nDetailed Results:")
    for test_result in results['test_results']:
        print(f"\nTest: {test_result['test_name']}")
        if "error" in test_result:
            print(f"  ERROR: {test_result['error']}")
        else:
            print(f"  Accuracy: {test_result['accuracy']:.1%}")
            print(f"  Documents: {test_result['identified_count']} identified, {test_result['expected_count']} expected")
            print(f"  Validation Score: {test_result['validation'].confidence_score:.2f}")
            if test_result['missing_types']:
                print(f"  Missing: {test_result['missing_types']}")
            if test_result['extra_types']:
                print(f"  Extra: {test_result['extra_types']}")

    print("\n" + "="*80)
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_validation_demo())